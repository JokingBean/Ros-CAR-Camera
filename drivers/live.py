#!/usr/bin/env python3
"""
PC 端连续定位 — 本地相机直连
===============================
三台 USB 相机持续抓帧 → Tag 检测 → 实时显示 XY + FPS。
"""

import cv2, yaml, numpy as np, time, os
from collections import deque
from src.tracking import detect_cube_extrinsics, grid_snap, TARGET_IDS


def main():
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open("cfg/extrinsics.yaml", "r") as f:
        ext = yaml.safe_load(f)

    # 打开所有相机
    caps = {}
    cam_params = {}
    for c in config["cameras"]:
        name = c["name"]
        idx = int(c["device"])
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -10)
        cap.set(cv2.CAP_PROP_CONTRAST, 40)
        time.sleep(0.3)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            print(f"  {name} (idx={idx}): {frame.shape[1]}x{frame.shape[0]} mean={frame.mean():.0f}")
        caps[name] = cap

        cm = c["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        R = np.array(ext[name]["R"]) if name in ext else np.eye(3)
        t = np.array(ext[name]["t"]).reshape(3, 1) if name in ext else np.array([[0], [0], [1.5]])
        cam_params[name] = (K, dist, R, t)

    active = [n for n in caps]
    print(f"\n  {len(active)} cameras ready. Ctrl+C stop.\n")

    fps_hist = deque(maxlen=30)
    pos_hist = deque(maxlen=5)
    t0 = time.time()

    try:
        while True:
            frames = {}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame

            all_results = []
            for name, frame in frames.items():
                if name in cam_params:
                    K, dist, R, t = cam_params[name]
                    results = detect_cube_extrinsics(frame, K, dist, R, t)
                    all_results.extend([(name, r) for r in results])

            t_now = time.time()
            elapsed = (t_now - t0) * 1000
            if elapsed > 0:
                fps_hist.append(1000 / elapsed)
            t0 = t_now
            fps = np.mean(fps_hist) if fps_hist else 0

            if all_results:
                good = [r for r in all_results if r[1].get("margin", 0) >= 20] or all_results
                xys = np.array([r["center_xy"] for n, r in good])
                gsds = np.array([r.get("gsd", 1.0) for n, r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused_xy = np.average(xys, axis=0, weights=w)
                pos_hist.append(fused_xy)
                smooth = np.mean(np.array(pos_hist), axis=0)
                gx, gy = grid_snap(smooth[0], smooth[1])
                err = np.linalg.norm([smooth[0] - gx, smooth[1] - gy]) * 100

                tags = " ".join(f"{n}T{r['tag_id']}" for n, r in good)
                print(f"\r  XY=({smooth[0]:.3f},{smooth[1]:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                      f"FPS={fps:.1f}  [{len(good)}/{len(all_results)}: {tags}]     ",
                      end="", flush=True)
            else:
                print(f"\r  等待立方体...  FPS={fps:.1f}                              ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n  停止")
    finally:
        for c in caps.values():
            c.release()


if __name__ == "__main__":
    main()
