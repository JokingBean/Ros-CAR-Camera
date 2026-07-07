"""
Pi 端 Triple-USB 检测服务 (v2 重写)
====================================
三台 USB 相机同时打开，统一曝光后持续检测地面 Tag + 立方体 Tag，
结果 JSON 通过 TCP 发送。不再依赖旧代码。
"""

import socket, json, struct, time, sys, os
import cv2, numpy as np
from pupil_apriltags import Detector

# ---------- 常量 ----------
WIDTH, HEIGHT = 1280, 720   # 检测分辨率（内参会自动缩放）
PORT = 9998
TAG_SIZE = 0.09            # 地面 Tag (米)
CUBE_TAG_SIZE = 0.135      # 立方体 Tag (米)
CUBE_IDS = {0, 1, 2, 3}

# ---------- 相机参数 (硬编码, PC 端注入) ----------
# 格式: {name: {idx, K, dist, R, t}}
CAMERAS = {}  # 由 PC 端启动前注入


def open_camera(idx, name):
    """打开一个 USB 相机，等自动曝光稳定后锁死参数。"""
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"{name} (video{idx}) 无法打开")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # 等自动曝光稳定，逐帧读到有画面
    time.sleep(0.5)
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame.mean() > 10:
            break
    else:
        # 30 帧后没画面 → 最后读一次
        ret, frame = cap.read()

    if not ret or frame.mean() < 5:
        cap.release()
        raise RuntimeError(f"{name} 无画面")

    # 确认有画面（最多读 10 次）

    return cap


def detect_floor_tags(img, detector, clahe, floor_map, tag_size):
    """检测地面 AprilTag，返回 homography (3x3) 或 None。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    dets = detector.detect(gray)
    pts = {}
    for d in dets:
        if d.tag_id in floor_map:
            pts[d.tag_id] = (d.center[0], d.center[1])
    if len(pts) < 4:
        return None
    world = np.array([floor_map[tid][:2] for tid in pts], dtype=np.float64)
    image = np.array([pts[tid] for tid in pts], dtype=np.float64)
    H, _ = cv2.findHomography(world, image, cv2.RANSAC, 3.0)
    return H


def detect_cube_tags(img, detector, clahe, K, dist, R, t):
    """检测立方体 Tag (ID 0-3), solvePnP 求 3D 位置。"""
    half = CUBE_TAG_SIZE / 2
    obj_pts = np.array([[-half, -half, 0], [half, -half, 0],
                         [half, half, 0], [-half, half, 0]], dtype=np.float64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    dets = detector.detect(gray)

    R_c2w = R.T
    t_c2w = -R_c2w @ t.reshape(3, 1)

    results = []
    for d in dets:
        if d.tag_id not in CUBE_IDS:
            continue
        ok, rv, tv = cv2.solvePnP(obj_pts, d.corners, K, dist)
        if not ok:
            continue
        Rt, _ = cv2.Rodrigues(rv)
        tw = (R_c2w @ tv.reshape(3, 1) + t_c2w).flatten()
        P = R @ tw.reshape(3, 1) + t.reshape(3, 1)
        gsd = float(np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000)
        results.append({
            "camera": "",  # 由外层填入
            "tag_id": int(d.tag_id),
            "xy": [float(tw[0]), float(tw[1])],
            "gsd": round(gsd, 2),
            "diag": float(np.linalg.norm(d.corners[0] - d.corners[2])),
            "margin": float(d.decision_margin),
        })
    return results


def main():
    if not CAMERAS:
        print("ERROR: CAMERAS config not injected", flush=True)
        sys.exit(1)

    # 打开所有相机
    print(f"[init] opening {len(CAMERAS)} cameras...", flush=True)
    caps = {}
    for name in CAMERAS:
        idx = CAMERAS[name]["idx"]
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  {name} (video{idx}): 无法打开", flush=True)
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        caps[name] = cap
        print(f"  {name} (video{idx}): OK", flush=True)

    # 统一预热
    if caps:
        print("[init] warming up...", flush=True)
        time.sleep(1)
        for _ in range(20):
            for cap in caps.values():
                cap.read()

        for name, cap in list(caps.items()):
            ret, frame = cap.read()
            if not ret or frame.mean() < 5:
                print(f"  {name}: warmup failed", flush=True)
                cap.release()
                del caps[name]
            else:
                print(f"  {name}: ready mean={frame.mean():.0f}", flush=True)

    if not caps:
        print("[init] no cameras", flush=True)
        sys.exit(1)

    # 预创建检测器
    floor_detector = Detector(families="tag36h11", quad_decimate=1.0)
    cube_detector = Detector(families="tag36h11", quad_decimate=1.0)
    floor_clahe = cv2.createCLAHE(2.0, (8, 8))
    cube_clahe = cv2.createCLAHE(2.0, (8, 8))

    # 收集地面 Tag 映射
    global _FLOOR_MAP
    _FLOOR_MAP = {int(k): (v["x"], v["y"], v["z"])
                  for k, v in _FLOOR_TAGS["tags"].items()}

    # TCP 服务
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT))
    sock.listen(1)
    print(f"[init] TCP listening on :{PORT}", flush=True)

    conn, addr = sock.accept()
    print(f"[run] connected from {addr}", flush=True)

    try:
        while True:
            t0 = time.time()
            batch = []

            for name, cap in caps.items():
                cfg = CAMERAS[name]
                ret, frame = cap.read()
                if not ret:
                    continue

                K = np.array(cfg["K"], dtype=np.float64)
                dist = np.array(cfg["dist"], dtype=np.float64)
                R = np.array(cfg["R"], dtype=np.float64)
                t = np.array(cfg["t"], dtype=np.float64)

                tags = detect_cube_tags(frame, cube_detector, cube_clahe, K, dist, R, t)
                for tag in tags:
                    tag["camera"] = name
                batch.extend(tags)

            data = json.dumps(batch).encode()
            conn.sendall(struct.pack(">I", len(data)) + data)

            elapsed = (time.time() - t0) * 1000
            if elapsed > 0:
                fps = 1000 / elapsed
                if len(batch) > 0:
                    print(f"\r  FPS={fps:.1f}  tags={len(batch)}", end="", flush=True)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[err] {e}", flush=True)
    finally:
        for cap in caps.values():
            cap.release()
        conn.close()
        sock.close()
        print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
