"""
Pi 端检测服务 — 持续抓帧 → Tag 检测 → JSON 推送
=================================================
TCP 端口 9998，启动后等待 PC 连接。
每循环检测三台相机，发送 JSON 结果。
"""
import socket, json, struct, cv2, numpy as np, time, sys
from pupil_apriltags import Detector

W, H = 2560, 1440
PORT = 9998
TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.135

# --- 从命令行参数或默认值加载配置 ---
# 这里先硬编码，PC 端会在启动前用 SFTP 覆盖实际配置
CAMERAS = {}  # 会被覆盖


def main():
    detector = Detector(families="tag36h11", quad_decimate=1.0)
    obj_pts = np.array([[-TAG_SIZE/2, -TAG_SIZE/2, 0], [TAG_SIZE/2, -TAG_SIZE/2, 0],
                         [TAG_SIZE/2, TAG_SIZE/2, 0], [-TAG_SIZE/2, TAG_SIZE/2, 0]],
                        dtype=np.float64)

    # 打开相机
    caps = {}
    for name, cfg in sorted(CAMERAS.items()):
        print(f"[init] opening {name} idx={cfg['idx']}", flush=True)
        cap = cv2.VideoCapture(cfg["idx"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        cap.set(cv2.CAP_PROP_CONTRAST, 40)
        cap.set(cv2.CAP_PROP_GAMMA, 100)
        time.sleep(0.2)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            print(f"[init] {name}: OK {frame.shape[1]}x{frame.shape[0]} mean={int(frame.mean())}", flush=True)
        else:
            print(f"[init] {name}: FAILED", flush=True)
        caps[name] = cap

    print(f"[init] {len(caps)} cameras ready, binding TCP:{PORT}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print(f"[run] connected from {addr}", flush=True)

    cycle = 0
    try:
        while True:
            t0 = time.time()
            results = []

            for name, cfg in CAMERAS.items():
                cap = caps.get(name)
                if cap is None:
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_s = cv2.resize(gray, None, fx=0.5, fy=0.5)
                gray_s = cv2.createCLAHE(2.0, (8, 8)).apply(gray_s)
                dets = detector.detect(gray_s)

                K = np.array(cfg["K"], dtype=np.float64)
                dist = np.array(cfg["dist"], dtype=np.float64)
                R_w2c = np.array(cfg["R"], dtype=np.float64)
                t_w2c = np.array(cfg["t"], dtype=np.float64).reshape(3, 1)
                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c

                for d in dets:
                    d.corners /= 2.0
                    d.center = (d.center[0] / 2, d.center[1] / 2)
                    if d.tag_id not in TARGET_IDS:
                        continue
                    ok, rv, tv = cv2.solvePnP(obj_pts, d.corners, K, dist)
                    if not ok:
                        continue
                    Rt, _ = cv2.Rodrigues(rv)
                    tw = (R_c2w @ tv.reshape(3, 1) + t_c2w).flatten()
                    P = R_w2c @ tw.reshape(3, 1) + t_w2c
                    gsd = float(np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000)

                    results.append(dict(
                        camera=name, tag_id=int(d.tag_id),
                        xy=[float(tw[0]), float(tw[1])],
                        gsd=round(gsd, 2),
                        diag=float(np.linalg.norm(d.corners[0] - d.corners[2])),
                        margin=float(d.decision_margin),
                    ))

            data = json.dumps(results).encode()
            conn.sendall(struct.pack('>I', len(data)) + data)

            cycle += 1
            elapsed = (time.time() - t0) * 1000
            if cycle % 10 == 0:
                print(f"[run] cycle={cycle} ms={elapsed:.0f} tags={len(results)}", flush=True)

    except Exception as e:
        print(f"[err] {e}", flush=True)
    finally:
        for c in caps.values():
            c.release()
        conn.close()
        sock.close()


if __name__ == "__main__":
    main()
