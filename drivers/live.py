"""
PC 端连续定位 (v2)
===================
连接 Pi TCP 服务，接收 JSON 检测结果，融合显示 XY + FPS。
"""

import socket, struct, json, yaml, numpy as np, time, os, sys, paramiko
from collections import deque

PI_HOST = "100.126.101.5"
PI_PORT = 9998

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg")


def load_yaml(name):
    with open(os.path.join(CFG_DIR, name), "r") as f:
        return yaml.safe_load(f)


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError()
        data += chunk
    return data


def upload_and_run(config, ext_data, floor_tags):
    """上传 Pi 服务脚本并启动。"""
    # 构建 CAMERAS 配置
    cams = {}
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        scale_x = 1280 / c["resolution"][0]
        scale_y = 720 / c["resolution"][1]
        cams[name] = {
            "idx": int(c["device"]),
            "K": [[cm["fx"] * scale_x, 0, cm["cx"] * scale_x],
                  [0, cm["fy"] * scale_y, cm["cy"] * scale_y],
                  [0, 0, 1]],
            "dist": c["dist_coeffs"],
            "R": ext_data.get(name, {}).get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "t": ext_data.get(name, {}).get("t", [0, 0, 1.5]),
        }

    # 读取服务器模板，注入 CAMERAS 和 _FLOOR_TAGS
    server_path = os.path.join(os.path.dirname(__file__), "..", "src", "pi_server.py")
    with open(server_path, "r") as f:
        code = f.read()

    code = code.replace(
        "CAMERAS = {}  # 由 PC 端启动前注入",
        f"CAMERAS = {json.dumps(cams)}")
    code = code.replace(
        "global _FLOOR_MAP\n    _FLOOR_MAP",
        f"_FLOOR_TAGS = {json.dumps(floor_tags)}\n    _FLOOR_MAP")

    print("Uploading to Pi...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username="pi", password="alcht0", timeout=10)

    # 释放摄像头
    ssh.exec_command("sudo fuser -k /dev/video0 /dev/video2 /dev/video4 2>/dev/null; sleep 1", timeout=5)

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/pi_server.py", "w") as f:
        f.write(code)
    sftp.close()

    ssh.exec_command("pkill -9 -f pi_server.py 2>/dev/null; sleep 1; "
                     "setsid python3 -u /tmp/pi_server.py >/tmp/pi_server.log 2>&1 &", timeout=5)
    ssh.close()

    # 等待就绪
    print("Waiting for Pi server...")
    for i in range(30):
        time.sleep(0.5)
        try:
            s = socket.socket()
            s.settimeout(1)
            s.connect((PI_HOST, PI_PORT))
            s.close()
            print("Connected!\n")
            return True
        except:
            if i % 5 == 0:
                print(f"  retrying ({i * 0.5:.0f}s)...")
    return False


def grid_snap(x, y, step=0.5):
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(0.0, min(4.5, gx)), max(0.0, min(5.0, gy))


def main():
    config = load_yaml("config.yaml")
    ext_data = load_yaml("extrinsics.yaml")
    floor_tags = load_yaml("floor_tags.yaml")

    print("=" * 60)
    print("  连续定位 (Pi v2)")
    print("=" * 60)

    if not upload_and_run(config, ext_data, floor_tags):
        print("Pi server not ready")
        return

    sock = socket.socket()
    sock.connect((PI_HOST, PI_PORT))
    sock.settimeout(2)

    fps_hist = deque(maxlen=30)
    pos_hist = deque(maxlen=5)
    t0 = time.time()

    try:
        while True:
            try:
                n = struct.unpack(">I", recv_exact(sock, 4))[0]
                raw = json.loads(recv_exact(sock, n))
            except (socket.timeout, ConnectionError):
                continue

            t_now = time.time()
            elapsed = (t_now - t0) * 1000
            if elapsed > 0:
                fps_hist.append(1000 / elapsed)
            t0 = t_now
            fps = np.mean(fps_hist) if fps_hist else 0

            if raw:
                good = [r for r in raw if r.get("margin", 0) >= 20] or raw
                xys = np.array([r["xy"] for r in good])
                gsds = np.array([r.get("gsd", 1.0) for r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused = np.average(xys, axis=0, weights=w)
                pos_hist.append(fused)
                smooth = np.mean(np.array(pos_hist), axis=0)
                gx, gy = grid_snap(smooth[0], smooth[1])
                err = np.linalg.norm([smooth[0] - gx, smooth[1] - gy]) * 100

                tags = " ".join(f"{r['camera']}T{r['tag_id']}" for r in good)
                print(f"\r  XY=({smooth[0]:.3f},{smooth[1]:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                      f"FPS={fps:.1f}  [{len(caps) if 'caps' in dir() else '?'}cam, {len(good)}/{len(raw)}: {tags}]     ",
                      end="", flush=True)
            else:
                print(f"\r  等待立方体...  FPS={fps:.1f}                           ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n  停止")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
