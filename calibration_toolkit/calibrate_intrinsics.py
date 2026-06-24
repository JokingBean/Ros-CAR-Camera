#!/usr/bin/env python3
"""
相机内参标定 — 棋盘格法
========================
支持 PiCamera (Picamera2) 和 USB 相机 (OpenCV)。

棋盘格: 9×9 内角点, 方格 1.8cm
采集要求: 15-25 张，多角度/多位置，棋盘格占画面 30%-70%
输出: camera_calibration_{name}.json

使用方法:
    python calibrate_intrinsics.py --camera picam    # PiCamera (1332×990)
    python calibrate_intrinsics.py --camera usb      # USB 相机 (1280×720)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ==============================================================
# 配置
# ==============================================================

CAMERA_CONFIGS = {
    "picam": {
        "name": "picam_1",
        "type": "picamera",
        "width": 1332,
        "height": 990,
        "fps": 60,
    },
    "usb": {
        "name": "usb_cam_1",
        "type": "usb",
        "width": 2048,
        "height": 1536,
        "fps": 30,
        "fourcc": "MJPG",      # USB 高分率必须用 MJPG
    },
}

CHESSBOARD = {
    "cols": 9,          # 内角点列数
    "rows": 9,          # 内角点行数
    "square_size": 0.020,  # 方格边长（米）= 2.0cm
}

MIN_IMAGES = 15         # 最少采集张数
RECOMMEND_IMAGES = 20   # 推荐张数
OUTPUT_DIR = "calibration_images"

# ==============================================================
# 相机操作
# ==============================================================

import platform
_IS_WINDOWS = platform.system() == "Windows"

def open_camera(cfg: dict):
    """根据配置打开相机，返回 (camera_obj, type_str)。"""
    if cfg["type"] == "picamera":
        from picamera2 import Picamera2
        cam = Picamera2(0)
        cam.configure(cam.create_video_configuration(
            main={"size": (cfg["width"], cfg["height"]), "format": "RGB888"},
            buffer_count=4,
            controls={"FrameDurationLimits": (16666, 16666)},
        ))
        cam.start()
        time.sleep(1.0)
        return cam, "picamera"
    else:
        # Windows: DSHOW, Linux: V4L2
        backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_V4L2
        cap = cv2.VideoCapture(0, backend)
        fourcc = cv2.VideoWriter_fourcc(*cfg.get("fourcc", "MJPG"))
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        time.sleep(0.5)
        return cap, "usb"

def read_frame(cam_obj, cam_type: str):
    """读取一帧 (BGR)。"""
    if cam_type == "picamera":
        return cam_obj.capture_array()       # RGB888 配置下实际输出 BGR
    else:
        ret, frame = cam_obj.read()
        return frame if ret else None

def close_camera(cam_obj, cam_type: str):
    if cam_type == "picamera":
        cam_obj.stop()
        cam_obj.close()
    else:
        cam_obj.release()

# ==============================================================
# 角点检测
# ==============================================================

def find_corners(gray, cols, rows):
    """检测棋盘格角点并亚像素精细化。"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

# ==============================================================
# 采集阶段
# ==============================================================

def capture_images(cfg: dict, chess: dict):
    """交互式采集标定图像。返回 (object_points_list, image_points_list, image_size)。"""
    print("\n" + "=" * 60)
    print(f"内参标定采集 — {cfg['name']} ({cfg['width']}×{cfg['height']})")
    print("=" * 60)
    print(f"棋盘格: {chess['cols']}×{chess['rows']}  方格 {chess['square_size']*100:.1f}cm")
    print()
    print("操作说明:")
    print("  将棋盘格在不同位置、不同角度展示给相机")
    print("  棋盘格应占画面 30%-70%")
    print("  包含: 平移、旋转、倾斜、靠近边缘等")
    print(f"  's' = 保存当前帧    'q' = 结束采集")
    print(f"  目标: ≥{MIN_IMAGES} 张（推荐 {RECOMMEND_IMAGES} 张）")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cam, cam_type = open_camera(cfg)
    w, h = cfg["width"], cfg["height"]
    cols, rows = chess["cols"], chess["rows"]

    # 世界坐标 (z=0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * chess["square_size"]

    saved = []

    window = f"Intrinsic Calibration — {cfg['name']}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # 窗口缩放，适应屏幕
    scale = min(900 / w, 700 / h, 1.0)
    cv2.resizeWindow(window, int(w * scale), int(h * scale))

    print("\n开始采集...")

    try:
        while True:
            frame = read_frame(cam, cam_type)
            if frame is None:
                continue

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners_refined = find_corners(gray, cols, rows)

            if ret:
                cv2.drawChessboardCorners(display, (cols, rows), corners_refined, ret)

                # 计算占比
                xs = corners_refined[:, 0, 0]
                ys = corners_refined[:, 0, 1]
                area_ratio = ((xs.max() - xs.min()) * (ys.max() - ys.min())
                              / (w * h) * 100)

                color = (0, 255, 0) if 20 < area_ratio < 80 else (0, 200, 255)
                cv2.putText(display, f"OK {area_ratio:.0f}%", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display, "No Chessboard", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 状态栏
            pct = len(saved) / RECOMMEND_IMAGES * 100
            status_color = (0, 255, 0) if len(saved) >= MIN_IMAGES else (0, 200, 255)
            bar_w = int(min(pct, 100) * 4)
            cv2.rectangle(display, (10, h - 40), (10 + bar_w, h - 15), status_color, -1)
            cv2.rectangle(display, (10, h - 40), (410, h - 15), (255, 255, 255), 1)
            cv2.putText(display, f"{len(saved)}/{RECOMMEND_IMAGES}",
                        (420, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and ret:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{OUTPUT_DIR}/{cfg['name']}_{ts}_{len(saved):03d}.jpg"
                cv2.imwrite(fname, frame)
                saved.append((objp.copy(), corners_refined.copy()))
                print(f"  [{len(saved)}] {fname}")
                cv2.waitKey(200)     # 防重复保存
            elif key == ord('q'):
                break

    finally:
        close_camera(cam, cam_type)
        cv2.destroyAllWindows()

    print(f"\n采集完成: {len(saved)} 张")

    if len(saved) < MIN_IMAGES:
        print(f"[警告] 仅 {len(saved)} 张，建议至少 {MIN_IMAGES} 张。标定精度可能不足。")

    objpoints = [s[0] for s in saved]
    imgpoints = [s[1] for s in saved]
    return objpoints, imgpoints, (w, h)

# ==============================================================
# 标定计算
# ==============================================================

def run_calibration(objpoints, imgpoints, image_size, chess: dict):
    """运行 calibrateCamera，返回结果字典或 None。"""
    print("\n" + "=" * 60)
    print("标定计算中...")
    print("=" * 60)

    if len(objpoints) < 5:
        print("[错误] 有效图像不足 (<5)")
        return None

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    if not ret:
        print("[错误] calibrateCamera 失败")
        return None

    # 重投影误差
    total_err = 0.0
    per_image_errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += e
        per_image_errs.append(float(e))
    mean_err = total_err / len(objpoints)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ratio = fy / fx

    print(f"\n标定结果:")
    print(f"  fx = {fx:.2f}    fy = {fy:.2f}    ratio = {ratio:.4f}")
    print(f"  cx = {cx:.2f}    cy = {cy:.2f}")
    print(f"  畸变: k1={D[0,0]:.5f}  k2={D[0,1]:.5f}  p1={D[0,2]:.5f}  p2={D[0,3]:.5f}")
    if D.shape[1] > 4:
        print(f"        k3={D[0,4]:.5f}")
    print(f"  平均重投影误差: {mean_err:.4f} px")
    print(f"  图像数: {len(objpoints)}")

    # 合理性检查
    warnings = []
    if ratio < 0.90 or ratio > 1.10:
        warnings.append(f"fy/fx = {ratio:.3f} 偏离 1.0，棋盘格角度覆盖可能不够")
    if mean_err > 0.5:
        warnings.append(f"重投影误差 {mean_err:.3f}px 偏高，检查棋盘格质量和采集角度")
    if cx < 0.4 * image_size[0] or cx > 0.6 * image_size[0]:
        warnings.append(f"cx={cx:.1f} 偏离图像中心 {image_size[0]//2}")
    if cy < 0.4 * image_size[1] or cy > 0.6 * image_size[1]:
        warnings.append(f"cy={cy:.1f} 偏离图像中心 {image_size[1]//2}")

    if warnings:
        print("\n[警告]")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\n[结果正常] ✓")

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "chessboard": {
            "width": chess["cols"],
            "height": chess["rows"],
            "square_size_cm": chess["square_size"] * 100,
        },
        "camera_matrix": {
            "fx": float(fx), "fy": float(fy),
            "cx": float(cx), "cy": float(cy),
            "data": K.tolist(),
        },
        "distortion_coeffs": {
            "k1": float(D[0, 0]),
            "k2": float(D[0, 1]),
            "p1": float(D[0, 2]),
            "p2": float(D[0, 3]),
            "k3": float(D[0, 4]) if D.shape[1] > 4 else 0.0,
            "data": D.tolist()[0] if len(D.shape) > 1 else D.tolist(),
        },
        "reprojection_error": float(mean_err),
        "per_image_errors": per_image_errs,
        "num_images": len(objpoints),
    }

# ==============================================================
# 入口
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="相机内参标定")
    parser.add_argument("--camera", choices=["picam", "usb"], required=True,
                        help="选择相机: picam 或 usb")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 文件路径 (默认 camera_calibration_{camera}.json)")
    args = parser.parse_args()

    cfg = CAMERA_CONFIGS[args.camera]
    output_file = args.output or f"camera_calibration_{args.camera}.json"

    # 采集
    objpoints, imgpoints, img_size = capture_images(cfg, CHESSBOARD)

    if len(objpoints) < 5:
        print("[退出] 图像不足，无法标定")
        sys.exit(1)

    # 计算
    result = run_calibration(objpoints, imgpoints, img_size, CHESSBOARD)
    if result is None:
        sys.exit(1)

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[保存] {output_file}")

    # 打印 config.yaml 片段
    K = result["camera_matrix"]
    D = result["distortion_coeffs"]
    print(f"\n===== config.yaml 内参片段 =====")
    print(f"camera_matrix:")
    print(f"  fx: {K['fx']}")
    print(f"  fy: {K['fy']}")
    print(f"  cx: {K['cx']}")
    print(f"  cy: {K['cy']}")
    print(f"dist_coeffs: [{D['k1']}, {D['k2']}, {D['p1']}, {D['p2']}, {D['k3']}]")
    print(f"=================================")

if __name__ == "__main__":
    main()
