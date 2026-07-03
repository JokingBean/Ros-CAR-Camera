#!/usr/bin/env python3
"""
相机内参标定 — 棋盘格法
========================
每台 USB 相机独立标定，获得各自的内参矩阵和畸变系数。

用法:
  # 在 PC 上逐个连接相机标定
  python calibration_toolkit/calibrate_intrinsics.py --camera usb1 --device 0
  python calibration_toolkit/calibrate_intrinsics.py --camera usb2 --device 1
  python calibration_toolkit/calibrate_intrinsics.py --camera usb3 --device 2

  标定结果自动保存到 calibration_toolkit/camera_calibration_{相机名}.json

操作:
  - 手持 9×9 棋盘格(2cm方格) 在相机视野内移动
  - 按 's' 保存当前帧（棋盘格需全部检测到）
  - 目标: ≥15 张，推荐 20 张
  - 按 'q' 结束采集并计算
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np


# 棋盘格配置（可调整）
CHESSBOARD_COLS = 9        # 内角点列数
CHESSBOARD_ROWS = 9        # 内角点行数
SQUARE_SIZE_MM = 20.0      # 方格边长（mm）
SQUARE_SIZE_M = SQUARE_SIZE_MM / 1000.0

# 采集目标
MIN_IMAGES = 15
TARGET_IMAGES = 20

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_output_path(camera_name: str) -> str:
    """生成输出 JSON 路径: calibration_toolkit/camera_calibration_{name}.json"""
    return os.path.join(OUTPUT_DIR, f"camera_calibration_{camera_name}.json")


def create_object_points():
    """生成棋盘格世界坐标 (z=0)。"""
    objp = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def find_checkerboard(gray, cols=None, rows=None):
    """检测棋盘格角点，返回亚像素精度的角点。"""
    if cols is None:
        cols = CHESSBOARD_COLS
    if rows is None:
        rows = CHESSBOARD_ROWS

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def calibrate_camera(obj_pts_list, img_pts_list, image_size):
    """执行内参标定。"""
    if len(obj_pts_list) < 3:
        return None, None, None, None, None

    gray = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None)

    # 计算每张图像的重投影误差
    per_image_errors = []
    for i in range(len(obj_pts_list)):
        proj, _ = cv2.projectPoints(
            obj_pts_list[i], rvecs[i], tvecs[i], K, D)
        err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts_list[i], axis=1))
        per_image_errors.append(float(err))

    total_error = float(np.mean(per_image_errors))
    return ret, K, D, total_error, per_image_errors


def main():
    parser = argparse.ArgumentParser(description="相机内参标定 — 棋盘格法")
    parser.add_argument("--camera", required=True,
                        help="相机名称（如 usb1, usb2, usb3），用于输出文件名")
    parser.add_argument("--device", type=int, default=0,
                        help="相机设备索引 (默认 0)")
    parser.add_argument("--width", type=int, default=2560,
                        help="相机宽度 (默认 2560)")
    parser.add_argument("--height", type=int, default=1440,
                        help="相机高度 (默认 1440)")
    parser.add_argument("--checkerboard-cols", type=int, default=CHESSBOARD_COLS,
                        help=f"棋盘格内角点列数 (默认 {CHESSBOARD_COLS})")
    parser.add_argument("--checkerboard-rows", type=int, default=CHESSBOARD_ROWS,
                        help=f"棋盘格内角点行数 (默认 {CHESSBOARD_ROWS})")
    parser.add_argument("--square-size-mm", type=float, default=SQUARE_SIZE_MM,
                        help=f"棋盘格方格边长 mm (默认 {SQUARE_SIZE_MM})")
    args = parser.parse_args()

    cols = args.checkerboard_cols
    rows = args.checkerboard_rows
    square_size_m = args.square_size_mm / 1000.0

    # 准备
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_pts_list = []   # 世界坐标列表
    img_pts_list = []   # 图像坐标列表
    captured_images = []
    image_size = (args.width, args.height)
    output_path = create_output_path(args.camera)

    # 打开相机
    print(f"\n{'='*60}")
    print(f"  内参标定 — {args.camera}")
    print(f"  棋盘格: {cols}x{rows}, 方格 {args.square_size_mm:.0f}mm")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  目标: ≥{MIN_IMAGES} 张 (推荐 {TARGET_IMAGES})")
    print(f"{'='*60}")
    print("\n操作指南:")
    print("  1. 将棋盘格放在相机视野内（占画面 30%-70%）")
    print("  2. 在画面不同位置、不同角度移动棋盘格")
    print("  3. 按 's' 保存（需全部角点检测到）")
    print("  4. 按 'q' 结束并计算")
    print("  5. 按 'r' 删除上一张\n")

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    time.sleep(0.5)

    window_name = f"Intrinsic Calibration — {args.camera}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    scale = min(1000 / args.width, 750 / args.height, 1.0)
    cv2.resizeWindow(window_name, int(args.width * scale), int(args.height * scale))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[错误] 无法读取相机")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 检测棋盘格
            found, corners = find_checkerboard(gray, cols, rows)

            status_color = (0, 255, 0) if found else (0, 0, 255)
            status_text = f"Checkerboard: {'OK' if found else 'NO'}"
            count_text = f"Captured: {len(obj_pts_list)}/{TARGET_IMAGES}"

            if found:
                cv2.drawChessboardCorners(display, (cols, rows), corners, found)

                # 绘制坐标轴指示
                origin = tuple(corners[0].astype(int).ravel())
                x_end = tuple(corners[rows - 1].astype(int).ravel())
                y_end = tuple(corners[1].astype(int).ravel())

                cv2.circle(display, origin, 8, (0, 0, 255), -1)
                cv2.arrowedLine(display, origin, x_end, (255, 0, 0), 2, tipLength=0.3)
                cv2.arrowedLine(display, origin, y_end, (0, 255, 0), 2, tipLength=0.3)
                cv2.putText(display, "X", (x_end[0] + 8, x_end[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display, "Y", (y_end[0] - 8, y_end[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 覆盖信息
            cv2.putText(display, status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display, count_text, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, "[s]ave  [r]emove last  [q]uit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and found:
                obj_pts_list.append(objp)
                img_pts_list.append(corners.reshape(-1, 2))
                captured_images.append(frame.copy())
                print(f"  [{len(obj_pts_list)}/{TARGET_IMAGES}] 已保存")

            elif key == ord('r'):
                if obj_pts_list:
                    obj_pts_list.pop()
                    img_pts_list.pop()
                    captured_images.pop()
                    print(f"  [{len(obj_pts_list)}/{TARGET_IMAGES}] 已删除上一张")

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 检查数量
    if len(obj_pts_list) < MIN_IMAGES:
        print(f"\n[错误] 采集张数不足: {len(obj_pts_list)}/{MIN_IMAGES}")
        print(f"       请重新标定，至少采集 {MIN_IMAGES} 张")
        sys.exit(1)

    # 标定
    print(f"\n正在标定（{len(obj_pts_list)} 张图像）...")
    ret, K, D, total_error, per_image_errors = calibrate_camera(
        obj_pts_list, img_pts_list, image_size)

    if not ret:
        print("[错误] 标定失败")
        sys.exit(1)

    # 输出结果
    print(f"\n{'='*60}")
    print(f"  标定结果 — {args.camera}")
    print(f"{'='*60}")
    print(f"  相机矩阵:")
    print(f"    fx = {K[0,0]:.4f}")
    print(f"    fy = {K[1,1]:.4f}")
    print(f"    cx = {K[0,2]:.4f}")
    print(f"    cy = {K[1,2]:.4f}")
    print(f"  畸变系数:")
    print(f"    k1 = {D[0,0]:.6f}")
    print(f"    k2 = {D[0,1]:.6f}")
    print(f"    p1 = {D[0,2]:.6f}")
    print(f"    p2 = {D[0,3]:.6f}")
    print(f"    k3 = {D[0,4]:.6f}")
    print(f"  重投影误差: {total_error:.4f} px")
    print(f"  {'良好' if total_error < 0.5 else '偏大（建议重标）' if total_error > 1.0 else '可接受'}")
    print(f"  图像数量: {len(obj_pts_list)}")

    # 保存 JSON
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_name": args.camera,
        "image_size": {"width": args.width, "height": args.height},
        "chessboard": {
            "width": cols, "height": rows,
            "square_size_m": square_size_m,
        },
        "camera_matrix": {
            "fx": K[0,0], "fy": K[1,1], "cx": K[0,2], "cy": K[1,2],
            "data": K.tolist(),
        },
        "distortion_coeffs": {
            "k1": float(D[0,0]), "k2": float(D[0,1]),
            "p1": float(D[0,2]), "p2": float(D[0,3]),
            "k3": float(D[0,4]),
            "data": D.flatten().tolist(),
        },
        "reprojection_error": float(total_error),
        "per_image_errors": per_image_errors,
        "num_images": len(obj_pts_list),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[保存] {output_path}")

    # 保存最后的预览图像（用于参考）
    preview_path = os.path.join(OUTPUT_DIR, f"calibration_preview_{args.camera}.jpg")
    if captured_images:
        # 在最后一张图上画角点
        preview = captured_images[-1].copy()
        gray_p = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        _, last_corners = find_checkerboard(gray_p, cols, rows)
        if last_corners is not None:
            cv2.drawChessboardCorners(preview, (cols, rows), last_corners, True)
        cv2.imwrite(preview_path, preview)

    print(f"\n{'='*60}")
    print(f"  下一步:")
    print(f"  将标定结果填入主项目 config.yaml 中 {args.camera} 的")
    print(f"  camera_matrix 和 dist_coeffs 字段")
    print(f"{'='*60}")

    # 打印 YAML 格式供直接复制
    print(f"\nYAML 配置片段（可直接复制到 config.yaml）:")
    print(f"  # --- {args.camera} ---")
    print(f"  camera_matrix:")
    print(f"    fx: {K[0,0]:.4f}")
    print(f"    fy: {K[1,1]:.4f}")
    print(f"    cx: {K[0,2]:.4f}")
    print(f"    cy: {K[1,2]:.4f}")
    print(f"  dist_coeffs: [{D[0,0]:.6f}, {D[0,1]:.6f}, {D[0,2]:.6f}, {D[0,3]:.6f}, {D[0,4]:.6f}]")


if __name__ == "__main__":
    main()
