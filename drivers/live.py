#!/usr/bin/env python3
"""
PC 端 — 连续定位。先尝试直连 Pi，失败则自动启动服务。
"""

import socket, json, struct, yaml, numpy as np, time, os, sys, paramiko
from collections import deque

from src.tracking import grid_snap

PI_HOST = "100.126.101.5"
PI_PORT = 9998
PI_SCRIPT = "/home/pi/UwbCamera/detect_server.py"


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("disconnected")
        data += chunk
    return data


def ensure_pi_server(config, ext_data):
    """确保 Pi 端检测服务在运行。"""
    import json as _json

    # 先尝试直连
    try:
        test = socket.socket()
        test.settimeout(2)
        test.connect((PI_HOST, PI_PORT))
        test.close()
        return  # 已经在跑
    except:
        pass

    print("  Pi server not running, starting...")

    # 构建相机配置
    cam_configs = {}
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        cam_configs[name] = {
            "idx": int(c["device"]),
            "K": [[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]],
            "dist": c["dist_coeffs"],
            "R": ext_data.get(name, {}).get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "t": ext_data.get(name, {}).get("t", [0, 0, 1.5]),
        }

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username="pi", password="alcht0", timeout=10)

    # 上传脚本
    ssh.exec_command("mkdir -p /home/pi/UwbCamera", timeout=5)
    with open("src/pi_detect_server.py", "r", encoding="utf-8") as f:
        server_code = f.read()
    server_code = server_code.replace(
        "CAMERAS = {}  # 会被覆盖",
        f"CAMERAS = {_json.dumps(cam_configs)}")
    sftp = ssh.open_sftp()
    with sftp.file(PI_SCRIPT, "w") as f:
        f.write(server_code)
    sftp.close()

    # 清摄像头 + 杀旧进程 + 启动
    ssh.exec_command(
        "for d in /dev/video0 /dev/video2 /dev/video4; do sudo fuser -k $d 2>/dev/null; done; "
        "sleep 1; pkill -9 -f detect_server.py 2>/dev/null; sleep 0.5; "
        f"nohup python3 -u {PI_SCRIPT} >/home/pi/UwbCamera/detect.log 2>&1 &",
        timeout=8)

    # 等 TCP 端口
    for i in range(40):
        time.sleep(0.5)
        try:
            test = socket.socket()
            test.settimeout(1)
            test.connect((PI_HOST, PI_PORT))
            test.close()
            print(f"  Server ready ({i*0.5:.0f}s)")
            break
        except:
            pass
    else:
        # 查日志找原因
        try:
            stdin, stdout, _ = ssh.exec_command("tail -3 /home/pi/UwbCamera/detect.log", timeout=5)
            print("  Log: " + stdout.read().decode().strip())
        except:
            pass
        raise RuntimeError("Pi server failed to start")

    ssh.close()


def main():
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open("cfg/extrinsics.yaml", "r") as f:
        ext_data = yaml.safe_load(f)

    ensure_pi_server(config, ext_data)

    # TCP 接收
    sock = socket.socket()
    sock.settimeout(2)
    sock.connect((PI_HOST, PI_PORT))

    fps_history = deque(maxlen=30)
    pos_history = deque(maxlen=5)
    t0 = time.time()

    print("=" * 60)
    print("  连续定位  Ctrl+C 停止")
    print("=" * 60)

    try:
        while True:
            try:
                n = struct.unpack('>I', recv_exact(sock, 4))[0]
                jdata = recv_exact(sock, n)
                raw = json.loads(jdata)
            except (socket.timeout, ConnectionError):
                print(f"\r  ...", end="", flush=True)
                continue

            t_now = time.time()
            t_elapsed = (t_now - t0) * 1000
            if t_elapsed > 0:
                fps_history.append(1000 / t_elapsed)
            t0 = t_now
            avg_fps = np.mean(fps_history) if fps_history else 0

            if raw:
                good = [r for r in raw if r.get("margin", 0) >= 20] or raw
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
                      f"FPS={avg_fps:.1f}  [{len(good)}/{len(raw)}: {tags_str}]     ",
                      end="", flush=True)
            else:
                print(f"\r  等待立方体...  FPS={avg_fps:.1f}                         ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n  停止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
