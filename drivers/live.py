#!/usr/bin/env python3
"""
PC 端 — TCP 接收 Pi 检测结果，实时融合显示
Pi 端本地做 Tag 检测，只传 JSON 结果。
"""

import socket, json, struct, yaml, numpy as np, time, os, sys, paramiko
from collections import deque

from src.tracking import grid_snap

PI_HOST = "100.126.101.5"
PI_PORT = 9998


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("disconnected")
        data += chunk
    return data


def main():
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open("cfg/extrinsics.yaml", "r") as f:
        ext_data = yaml.safe_load(f)

    # 上传 Pi 端检测服务
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username="pi", password="alcht0", timeout=10)

    import json as _json

    cam_configs = {}
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        cam_configs[name] = {
            "idx": int(c["device"]),
            "K": [[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]],
            "dist": c["dist_coeffs"],
            "R": ext_data.get(name, dict()).get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "t": ext_data.get(name, dict()).get("t", [0, 0, 1.5]),
        }

    import json as _json
    cam_configs_str = _json.dumps(cam_configs)

    server_code = f"""import socket, json, struct, cv2, numpy as np, time
from pupil_apriltags import Detector

W, H = 2560, 1440
TARGET_IDS = {{0, 1, 2, 3}}
TAG_SIZE = 0.135
CAMERAS = {cam_configs_str}

detector = Detector(families='tag36h11', quad_decimate=1.0)
obj_pts = np.array([[-TAG_SIZE/2,-TAG_SIZE/2,0],[TAG_SIZE/2,-TAG_SIZE/2,0],
                     [TAG_SIZE/2,TAG_SIZE/2,0],[-TAG_SIZE/2,TAG_SIZE/2,0]], dtype=np.float64)

caps = {{}}
for name, cfg in CAMERAS.items():
    cap = cv2.VideoCapture(cfg['idx'], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    time.sleep(0.3)
    for _ in range(10): cap.read()
    caps[name] = cap

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', {PI_PORT}))
sock.listen(1)
conn, addr = sock.accept()

try:
    while True:
        results = []
        for name, cfg in CAMERAS.items():
            ret, frame = caps[name].read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_s = cv2.resize(gray, None, fx=0.5, fy=0.5)
            gray_s = cv2.createCLAHE(2.0,(8,8)).apply(gray_s)
            dets = detector.detect(gray_s)
            K = np.array(cfg['K'], dtype=np.float64)
            dist = np.array(cfg['dist'], dtype=np.float64)
            R_w2c = np.array(cfg['R'], dtype=np.float64)
            t_w2c = np.array(cfg['t'], dtype=np.float64).reshape(3,1)
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            for d in dets:
                d.corners /= 2.0
                d.center = (d.center[0]/2, d.center[1]/2)
                if d.tag_id not in TARGET_IDS:
                    continue
                ok, rv, tv = cv2.solvePnP(obj_pts, d.corners, K, dist)
                if not ok:
                    continue
                Rt, _ = cv2.Rodrigues(rv)
                tw = (R_c2w @ tv.reshape(3,1) + t_c2w).flatten()
                P = R_w2c @ tw.reshape(3,1) + t_w2c
                gsd = float(np.linalg.norm(P) / ((K[0,0]+K[1,1])/2) * 1000)
                results.append(dict(
                    camera=name, tag_id=int(d.tag_id),
                    xy=[float(tw[0]), float(tw[1])],
                    gsd=round(gsd, 2),
                    diag=float(np.linalg.norm(d.corners[0]-d.corners[2])),
                    margin=float(d.decision_margin),
                ))
        data = json.dumps(results).encode()
        conn.sendall(struct.pack('>I', len(data)) + data)
except:
    pass
finally:
    for c in caps.values(): c.release()
    conn.close()
    sock.close()
"""

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/detect_server.py", "w") as f:
        f.write(server_code)
    sftp.close()

    ssh.exec_command(
        f"sudo killall -9 python3 2>/dev/null; sleep 2; "
        f"nohup python3 /tmp/detect_server.py >/tmp/detect.log 2>&1 &")
    ssh.close()

    # 等待 Pi 服务就绪
    print("Starting Pi detection server...")
    for _ in range(30):
        time.sleep(0.5)
        try:
            test = socket.socket()
            test.settimeout(1)
            test.connect((PI_HOST, PI_PORT))
            test.close()
            print("Connected. Ctrl+C stop.\n")
            break
        except:
            pass
    else:
        print("Pi server not ready")
        return

    # 连接 TCP，接收 JSON
    sock = socket.socket()
    sock.settimeout(5)
    sock.connect((PI_HOST, PI_PORT))
    sock.settimeout(1)

    fps_history = deque(maxlen=30)
    pos_history = deque(maxlen=5)

    try:
        while True:
            t0 = time.time()

            try:
                n = struct.unpack('>I', recv_exact(sock, 4))[0]
                jdata = recv_exact(sock, n)
                all_results = json.loads(jdata)
            except (socket.timeout, ConnectionError):
                continue

            t_elapsed = (time.time() - t0) * 1000
            fps_history.append(1000 / t_elapsed if t_elapsed > 0 else 0)
            avg_fps = np.mean(fps_history) if fps_history else 0

            if all_results:
                good = [r for r in all_results if r.get("margin", 0) >= 20] or all_results
                xys = np.array([r["xy"] for r in good])
                gsds = np.array([r.get("gsd", 1.0) for r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused_xy = np.average(xys, axis=0, weights=w)
                pos_history.append(fused_xy)
                smooth_xy = np.mean(np.array(pos_history), axis=0)
                gx, gy = grid_snap(smooth_xy[0], smooth_xy[1])
                err = np.linalg.norm([smooth_xy[0] - gx, smooth_xy[1] - gy]) * 100

                tags_str = " ".join(f"{r['camera']}T{r['tag_id']}" for r in good)
                print(f"\r  XY=({smooth_xy[0]:.3f},{smooth_xy[1]:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                      f"FPS={avg_fps:.1f}  [{len(good)} tags: {tags_str}]   ",
                      end="", flush=True)
            else:
                print(f"\r  未检测到  FPS={avg_fps:.1f}                         ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  停止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
