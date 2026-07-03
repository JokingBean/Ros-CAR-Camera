#!/usr/bin/env python3
"""
Pi 版本 — 三相机 BEV + 定位 + 连续追踪
=========================================
在树莓派上直接运行，不依赖 PC 端通信。

用法:
  python pi_bev.py          # BEV 俯视图
  python pi_bev.py --live   # 连续定位模式
  python pi_bev.py --test   # 精度逐点测试
"""

import cv2, yaml, numpy as np, time, os, sys, json
from datetime import datetime
from collections import deque
from pupil_apriltags import Detector

# ============================================================
# 配置
# ============================================================
W, H = 2560, 1440
TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.135
GRID_STEP = 0.5
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = 0.0, 5.0

# BEV 参数
BEV_PPM = 200
BEV_MARGIN = 50
BEV_W = int((X_MAX - X_MIN) * BEV_PPM) + 2 * BEV_MARGIN
BEV_H = int((Y_MAX - Y_MIN) * BEV_PPM) + 2 * BEV_MARGIN


# ============================================================
# 相机 + 外参加载
# ============================================================
def load_setup():
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

    return config, cam_params


def open_cameras(config):
    """打开所有相机，返回 {name: VideoCapture}"""
    caps = {}
    for c in config["cameras"]:
        name = c["name"]
        idx = int(c["device"])
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        cap.set(cv2.CAP_PROP_CONTRAST, 40)
        cap.set(cv2.CAP_PROP_GAMMA, 100)
        time.sleep(0.3)
        for _ in range(5):
            cap.read()
        caps[name] = cap
        print(f"  [{name}] opened")
    return caps


# ============================================================
# 检测
# ============================================================
def detect_cube(name, img, K, dist, R, t):
    """solvePnP 检测立方体 Tag"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.resize(gray, None, fx=0.5, fy=0.5)
    gray_s = cv2.createCLAHE(2.0, (8, 8)).apply(gray_s)

    dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray_s)
    results = []
    half = TAG_SIZE / 2.0
    obj_pts = np.array([[-half, -half, 0], [half, -half, 0],
                         [half, half, 0], [-half, half, 0]], dtype=np.float64)

    R_c2w = R.T
    t_c2w = -R_c2w @ t

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
        P = R @ tw.reshape(3, 1) + t
        gsd = float(np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000)
        results.append(dict(
            camera=name, tag_id=int(d.tag_id),
            xy=[float(tw[0]), float(tw[1])],
            gsd=round(gsd, 2),
            diag=float(np.linalg.norm(d.corners[0] - d.corners[2])),
            margin=float(d.decision_margin),
        ))
    return results


def grid_snap(x, y):
    gx = round(x / GRID_STEP) * GRID_STEP
    gy = round(y / GRID_STEP) * GRID_STEP
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))


# ============================================================
# BEV 生成 (简化版)
# ============================================================
def make_bev(img, K, R, t):
    """单相机 BEV 投影"""
    h_img, w_img = img.shape[:2]
    bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    step = 1.0 / BEV_PPM
    for bv in range(BEV_H):
        yw = Y_MAX - (bv - BEV_MARGIN) * step
        for bu in range(BEV_W):
            xw = X_MIN + (bu - BEV_MARGIN) * step
            Pw = np.array([[xw], [yw], [0.0]])
            Pc = R @ Pw + t
            if Pc[2, 0] <= 0:
                continue
            uv = K @ Pc
            u, v = int(uv[0, 0] / uv[2, 0]), int(uv[1, 0] / uv[2, 0])
            if 0 <= u < w_img and 0 <= v < h_img:
                bev[bv, bu] = img[v, u]
    return bev


# ============================================================
# 模式 1: BEV
# ============================================================
def mode_bev():
    config, cam_params = load_setup()
    caps = open_cameras(config)
    time.sleep(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"bev_{ts}", exist_ok=True)

    bevs = {}
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(f"bev_{ts}/{name}.jpg", frame)
        K, dist, R, t = cam_params[name]
        bevs[name] = make_bev(frame, K, R, t)
        print(f"  {name}: BEV done")

    if not bevs:
        print("no images")
        return

    # 融合
    fused = np.zeros((BEV_H, BEV_W, 3), dtype=np.float32)
    count = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    for name, bev in bevs.items():
        mask = (bev.sum(axis=2) > 0)
        fused[mask] += bev[mask].astype(np.float32)
        count[mask] += 1
    valid = count > 0
    fused[valid] = (fused[valid] / count[valid, None]).astype(np.uint8)

    cv2.imwrite(f"bev_{ts}/fused.jpg", fused)
    print(f"  saved to bev_{ts}/")

    for cap in caps.values():
        cap.release()


# ============================================================
# 模式 2: 连续定位
# ============================================================
def mode_live():
    config, cam_params = load_setup()
    caps = open_cameras(config)

    fps_hist = deque(maxlen=30)
    pos_hist = deque(maxlen=5)
    t0 = time.time()

    print("\n  Ctrl+C 停止\n")
    try:
        while True:
            all_results = []
            for name, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    continue
                K, dist, R, t = cam_params[name]
                results = detect_cube(name, frame, K, dist, R, t)
                all_results.extend(results)

            t_now = time.time()
            t_elapsed = (t_now - t0) * 1000
            if t_elapsed > 0:
                fps_hist.append(1000 / t_elapsed)
            t0 = t_now
            avg_fps = np.mean(fps_hist) if fps_hist else 0

            if all_results:
                good = [r for r in all_results if r.get("margin", 0) >= 20] or all_results
                xys = np.array([r["xy"] for r in good])
                gsds = np.array([r.get("gsd", 1.0) for r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused = np.average(xys, axis=0, weights=w)
                pos_hist.append(fused)
                smooth = np.mean(np.array(pos_hist), axis=0)
                gx, gy = grid_snap(smooth[0], smooth[1])
                err = np.linalg.norm([smooth[0] - gx, smooth[1] - gy]) * 100
                tags = " ".join(f"{r['camera'][-1]}T{r['tag_id']}" for r in good)
                print(f"\r  XY=({smooth[0]:.3f},{smooth[1]:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                      f"FPS={avg_fps:.1f}  [{len(good)}/{len(all_results)}: {tags}]   ",
                      end="", flush=True)
            else:
                print(f"\r  等待立方体...  FPS={avg_fps:.1f}                         ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n  停止")
    finally:
        for cap in caps.values():
            cap.release()


# ============================================================
# 模式 3: 精度点测
# ============================================================
def mode_test():
    config, cam_params = load_setup()
    caps = open_cameras(config)

    results_file = f"precision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    history = []

    print("\n  Enter 测量 | q 退出 | s 统计\n")
    while True:
        cmd = input("  > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            if history:
                errs = [m["error_cm"] for m in history]
                print(f"  {len(history)}次: avg={np.mean(errs):.1f}cm max={np.max(errs):.1f}cm")
            continue

        all_results = []
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            K, dist, R, t = cam_params[name]
            results = detect_cube(name, frame, K, dist, R, t)
            all_results.extend(results)

        if not all_results:
            print("  未检测到立方体")
            continue

        good = [r for r in all_results if r.get("margin", 0) >= 20] or all_results
        xys = np.array([r["xy"] for r in good])
        gsds = np.array([r.get("gsd", 1.0) for r in good])
        w = 1.0 / np.maximum(gsds, 0.01)
        w /= w.sum()
        fused = np.average(xys, axis=0, weights=w)
        gx, gy = grid_snap(fused[0], fused[1])
        err = np.linalg.norm([fused[0] - gx, fused[1] - gy]) * 100

        for r in all_results:
            print(f"    [{r['camera']}] T{r['tag_id']} xy=({r['xy'][0]:.3f},{r['xy'][1]:.3f}) gsd={r['gsd']}mm")
        print(f"  XY=({fused[0]:.3f},{fused[1]:.3f}) grid=({gx:.1f},{gy:.1f}) err={err:.1f}cm")

        record = {"time": datetime.now().strftime("%H%M%S"), "grid": [gx, gy],
                  "fused": [round(float(fused[0]), 3), round(float(fused[1]), 3)], "error_cm": round(float(err), 1)}
        history.append(record)
        with open(results_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    for cap in caps.values():
        cap.release()
    if history:
        print(f"\n  saved: {results_file}")


# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.live:
        mode_live()
    elif args.test:
        mode_test()
    else:
        mode_bev()
