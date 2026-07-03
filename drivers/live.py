#!/usr/bin/env python3
"""
PC 端 — 通过 TCP 接收 Pi 相机流，实时定位
"""

import socket, struct, cv2, yaml, numpy as np, time, os, sys
from collections import deque

from src.tracking import detect_cube_extrinsics, grid_snap, TARGET_IDS

PI_HOST = "100.126.101.5"
PI_PORT = 9998

W, H = 2560, 1440


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
        ext = yaml.safe_load(f)

    cam_params = {}
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        R = np.array(ext[name]["R"]) if name in ext else np.eye(3)
        t = np.array(ext[name]["t"]).reshape(3, 1) if name in ext else np.array([[0], [0], [1.5]])
        cam_params[name] = (K, dist, R, t)

    # 启动 Pi 端 TCP 流服务
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username="pi", password="alcht0", timeout=10)

    server_script = f"""
import socket, cv2, struct, time, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', {PI_PORT}))
sock.listen(1)
print('READY', flush=True)
conn, addr = sock.accept()
print(f'CONNECTED {{addr}}', flush=True)

caps = {{}}
for idx, name in [(0,'usb1'), (2,'usb2'), (4,'usb3')]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, {W})
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, {H})
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    time.sleep(0.3)
    for _ in range(10): cap.read()
    caps[name] = cap

try:
    while True:
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret: continue
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok: continue
            data = jpg.tobytes()
            name_bytes = name.encode()
            header = struct.pack('>I', len(name_bytes)) + name_bytes + struct.pack('>I', len(data))
            conn.sendall(header + data)
except:
    pass
finally:
    for c in caps.values(): c.release()
    conn.close()
    sock.close()
"""

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/stream.py", "w") as f:
        f.write(server_script)
    sftp.close()

    # Start TCP server in background
    ssh.exec_command(f"pkill -f stream.py 2>/dev/null; sleep 1; nohup python3 /tmp/stream.py >/tmp/stream.log 2>&1 &")
    ssh.close()

    # Wait for server to be ready
    print("Waiting for Pi TCP server...")
    for _ in range(10):
        time.sleep(0.5)
        try:
            test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test.settimeout(1)
            test.connect((PI_HOST, PI_PORT))
            test.close()
            print("Connected!")
            break
        except:
            pass
    else:
        print("Server not ready, check Pi")
        return

    # 连接 TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((PI_HOST, PI_PORT))
    sock.settimeout(1)
    print("Connected. Ctrl+C stop.\n")

    fps_history = deque(maxlen=30)
    pos_history = deque(maxlen=5)

    try:
        while True:
            t0 = time.time()

            frames = {}
            while True:
                try:
                    name_len = struct.unpack('>I', recv_exact(sock, 4))[0]
                    name = recv_exact(sock, name_len).decode()
                    jpg_len = struct.unpack('>I', recv_exact(sock, 4))[0]
                    jpg = recv_exact(sock, jpg_len)
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        frames[name] = img
                    if len(frames) >= 3:
                        break
                except (socket.timeout, ConnectionError):
                    break

            if not frames:
                continue

            # 检测 + 融合
            all_results = []
            for name, frame in frames.items():
                if name in cam_params:
                    K, dist, R, t = cam_params[name]
                    results = detect_cube_extrinsics(frame, K, dist, R, t)
                    all_results.extend([(name, r) for r in results])

            t_elapsed = (time.time() - t0) * 1000
            fps_history.append(1000 / t_elapsed if t_elapsed > 0 else 0)
            avg_fps = np.mean(fps_history) if fps_history else 0

            if all_results:
                good = [r for r in all_results if r[1].get("margin", 0) >= 20] or all_results
                xys = np.array([r[1]["center_xy"] for r in good])
                gsds = np.array([r[1].get("gsd", 1.0) for r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused_xy = np.average(xys, axis=0, weights=w)
                pos_history.append(fused_xy)
                smooth_xy = np.mean(np.array(pos_history), axis=0)
                gx, gy = grid_snap(smooth_xy[0], smooth_xy[1])
                err = np.linalg.norm([smooth_xy[0] - gx, smooth_xy[1] - gy]) * 100

                tags_str = " ".join(f"{n}T{r['tag_id']}" for n, r in good)
                print(f"\r  XY=({smooth_xy[0]:.3f},{smooth_xy[1]:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                      f"FPS={avg_fps:.1f}  [{len(good)} tags: {tags_str}]   ",
                      end="", flush=True)
            else:
                print(f"\r  未检测到  FPS={avg_fps:.1f}  {len(frames)} cameras    ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  停止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
