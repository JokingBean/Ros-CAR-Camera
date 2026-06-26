#!/usr/bin/env python3
"""
相机外参标定 — 棋盘格地面法 + AprilTag PnP 法
==============================================
支持两种标定模式：

  模式A (棋盘格): 棋盘格平放地面 → 计算单应性矩阵 H + 相机位姿
  模式B (AprilTag): 已知地面 Tag 位置 → PnP 求解 R, t

输出格式兼容:
  - 单应性矩阵: camera_extrinsic_{name}.json (与旧版 GroundTextureCamera 兼容)
  - 相机外参: extrinsics_{name}.yaml (与新版 ROS-Camera 多相机系统兼容)

使用方法:
  python calibrate_extrinsics.py --camera picam --mode chessboard
  python calibrate_extrinsics.py --camera picam --mode apriltag
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

_IS_WINDOWS = platform.system() == "Windows"

# ==============================================================
# 相机配置（与 calibrate_intrinsics.py 保持一致）
# ==============================================================

CAMERA_CONFIGS = {
    "picam": {
        "name": "picam_1",
        "type": "picamera",
        "width": 2028,
        "height": 1520,
        "fps": 50,
    },
    "usb": {
        "name": "usb_cam_1",
        "type": "usb",
        "device": 0,
        "width": 2048,
        "height": 1536,
        "fps": 30,
        "fourcc": "MJPG",
    },
    "usb2": {
        "name": "usb_cam_2",
        "type": "usb",
        "device": 1,
        "width": 2560,
        "height": 1440,
        "fps": 30,
        "fourcc": "MJPG",
    },
}

# ==============================================================
# 相机操作
# ==============================================================

def open_camera(cfg: dict):
    if cfg["type"] == "picamera":
        from picamera2 import Picamera2
        cam = Picamera2(0)
        cam.configure(cam.create_video_configuration(
            main={"size": (cfg["width"], cfg["height"]), "format": "RGB888"},
            buffer_count=4,
        ))
        cam.start()
        time.sleep(1.0)
        return cam, "picamera"
    else:
        backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_V4L2
        idx = cfg.get("device", 0)
        cap = cv2.VideoCapture(idx, backend)
        fourcc = cv2.VideoWriter_fourcc(*cfg.get("fourcc", "MJPG"))
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        time.sleep(0.5)
        return cap, "usb"

def read_frame(cam_obj, cam_type: str):
    if cam_type == "picamera":
        arr = cam_obj.capture_array()
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
# 加载内参
# ==============================================================

def load_intrinsics(path: str):
    """从 camera_calibration.json 加载内参。"""
    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64)
    D = np.array([calib["distortion_coeffs"]["data"]], dtype=np.float64)
    print(f"内参: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    return K, D

# ==============================================================
# 模式 A: 棋盘格地面法
# ==============================================================

def calibrate_chessboard(cfg: dict, K, D):
    """棋盘格平放地面 → 单应性矩阵 + PnP 位姿。"""
    print("\n" + "=" * 60)
    print(f"外参标定 [棋盘格模式] — {cfg['name']}")
    print("=" * 60)
    print("操作:")
    print("  1. 将棋盘格平放在相机视野内的地面上")
    print("  2. 棋盘格第一个角点为世界原点")
    print("  3. X 轴沿棋盘格行方向，Y 轴沿列方向")
    print("  4. 确认角点全部检测到后按 'c' 确认")
    print("  5. 按 'q' 退出")

    cols, rows = 9, 9
    square_size = 0.020  # 2cm

    # 世界坐标 (z=0, 平放在地)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    cam, cam_type = open_camera(cfg)
    window = f"Extrinsic [Chessboard] — {cfg['name']}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = min(900 / cfg["width"], 700 / cfg["height"], 1.0)
    cv2.resizeWindow(window, int(cfg["width"] * scale), int(cfg["height"] * scale))

    corners_detected = None
    corners_undistorted = None

    try:
        while True:
            frame = read_frame(cam, cam_type)
            if frame is None:
                continue
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)

            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners_detected = corners
                # 对角点去畸变
                corners_undistorted = cv2.undistortPoints(
                    corners, K, D, None, K).reshape(-1, 2)

                cv2.drawChessboardCorners(display, (cols, rows), corners, ret)

                # 标记原点 (0,0) 和坐标轴
                origin = tuple(corners[0].astype(int).ravel())
                cv2.circle(display, origin, 12, (0, 0, 255), -1)
                cv2.putText(display, "Origin(0,0)", (origin[0]+15, origin[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                x_end = tuple(corners[rows - 1].astype(int).ravel())
                y_end = tuple(corners[1].astype(int).ravel())
                cv2.arrowedLine(display, origin, x_end, (255, 0, 0), 3, tipLength=0.3)
                cv2.arrowedLine(display, origin, y_end, (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(display, "X", (x_end[0]+10, x_end[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(display, "Y", (y_end[0]-10, y_end[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(display, "Detected! Press 'c' to calibrate", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No Chessboard", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and corners_detected is not None:
                break
            elif key == ord('q'):
                print("已退出")
                return

    finally:
        close_camera(cam, cam_type)
        cv2.destroyAllWindows()

    # 计算单应性矩阵
    world_pts = objp[:, :2]
    img_pts = corners_undistorted

    H, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, 3.0)
    if H is None:
        print("[错误] 单应性矩阵计算失败")
        return

    H_inv = np.linalg.inv(H)

    # PnP 计算相机位姿
    ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners_undistorted, K, None)
    if ret_pnp:
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()
        height = abs(cam_pos[2])
        print(f"\n相机位姿:")
        print(f"  高度: {height:.3f}m ({height*100:.1f}cm)")
        print(f"  位置: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})m")
    else:
        R, tvec, cam_pos, height = None, None, None, None

    # 保存结果 (JSON, 兼容旧版 GroundTextureCamera)
    result_json = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Ground plane to image homography",
        "chessboard": {"width": cols, "height": rows, "square_size_m": square_size},
        "homography": {
            "H_world_to_image": H.tolist(),
            "H_image_to_world": H_inv.tolist(),
        },
        "camera_matrix": K.tolist(),
        "distortion_coeffs": D.tolist()[0] if len(D.shape) > 1 else D.tolist(),
    }
    if ret_pnp:
        result_json["rotation_vector"] = rvec.flatten().tolist()
        result_json["translation_vector"] = tvec.flatten().tolist()
        result_json["rotation_matrix"] = R.tolist()
        result_json["camera_height_m"] = float(height)

    json_path = f"camera_extrinsic_{cfg['name']}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    print(f"\n[保存] {json_path}")

    # 也保存 YAML 格式 (兼容新版 ROS-Camera)
    if ret_pnp:
        yaml_data = {cfg["name"]: {"R": R.tolist(), "t": tvec.flatten().tolist()}}
        yaml_path = f"extrinsics_{cfg['name']}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=None)
        print(f"[保存] {yaml_path}")


# ==============================================================
# 模式 B: AprilTag PnP 法
# ==============================================================

def calibrate_apriltag(cfg: dict, K, D):
    """利用已知地面 AprilTag 通过 PnP 求解外参。"""
    from pupil_apriltags import Detector

    # 检查 floor_tags.yaml
    if not os.path.exists("floor_tags.yaml"):
        print("[错误] 未找到 floor_tags.yaml，请确保从主项目目录运行")
        print("       或将 floor_tags.yaml 复制到当前目录")
        return

    with open("floor_tags.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    floor_tags = {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
                  for k, v in raw["tags"].items()}

    print("\n" + "=" * 60)
    print(f"外参标定 [AprilTag PnP 模式] — {cfg['name']}")
    print("=" * 60)
    print(f"已加载 {len(floor_tags)} 个地面 Tag")
    print()
    print("操作:")
    print("  确保相机能看到多个地面 AprilTag (≥6 个)")
    print("  按 'c' 采集当前帧并计算外参")
    print("  按 'q' 退出")

    detector = Detector(families="tag36h11", quad_decimate=1.0)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    tag_size = 0.09  # 地面 Tag 边长

    cam, cam_type = open_camera(cfg)
    window = f"Extrinsic [AprilTag] — {cfg['name']}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = min(900 / cfg["width"], 700 / cfg["height"], 1.0)
    cv2.resizeWindow(window, int(cfg["width"] * scale), int(cfg["height"] * scale))

    try:
        while True:
            frame = read_frame(cam, cam_type)
            if frame is None:
                continue
            display = frame.copy()
            gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # 大图降采样加速
            h, w = gray.shape
            if w > 1500:
                gray_detect = cv2.resize(gray, None, fx=0.5, fy=0.5)
            else:
                gray_detect = gray

            detections = detector.detect(gray_detect)
            # 如果降采样了，把坐标缩回
            if w > 1500:
                for d in detections:
                    d.corners *= 2.0
                    d.center = (d.center[0]*2.0, d.center[1]*2.0)

            # 统计地面 Tag
            floor_dets = [d for d in detections if d.tag_id in floor_tags]
            n_floor = len(floor_dets)

            for d in detections:
                pts = d.corners.astype(int)
                color = (0, 255, 0) if d.tag_id in floor_tags else (255, 100, 0)
                cv2.polylines(display, [pts], True, color, 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(display, f"#{d.tag_id}", (cx-15, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            status = (f"Floor tags: {n_floor}  "
                      f"(need ≥6)  [c]=calibrate  [q]=quit")
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and n_floor >= 6:
                break
            elif key == ord('q'):
                print("已退出")
                return

    finally:
        close_camera(cam, cam_type)
        cv2.destroyAllWindows()

    # PnP 标定
    print("\n计算外参...")
    half = tag_size / 2.0

    # 过滤画面边缘的 Tag（边缘畸变大，检测不准）
    iw, ih = cfg["width"], cfg["height"]
    margin = 0.08  # 画面边缘 8% 以内的 Tag 丢弃
    floor_dets_filtered = []
    for d in floor_dets:
        cx, cy = d.center
        if (margin * iw < cx < (1 - margin) * iw and
            margin * ih < cy < (1 - margin) * ih):
            floor_dets_filtered.append(d)
    skipped = len(floor_dets) - len(floor_dets_filtered)
    if skipped:
        print(f"  过滤边缘 Tag: {skipped} 个（画面边缘 {margin*100:.0f}% 内）")

    obj_pts, img_pts = [], []
    for d in floor_dets_filtered:
        wpt = floor_tags[d.tag_id]
        c3 = np.array([
            [wpt[0]-half, wpt[1]-half, wpt[2]],
            [wpt[0]+half, wpt[1]-half, wpt[2]],
            [wpt[0]+half, wpt[1]+half, wpt[2]],
            [wpt[0]-half, wpt[1]+half, wpt[2]],
        ], dtype=np.float64)
        for c3i, c2i in zip(c3, d.corners):
            obj_pts.append(c3i)
            img_pts.append(c2i)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    n_floor_filtered = len(floor_dets_filtered)
    print(f"  角点: {len(obj_pts)} (来自 {n_floor_filtered} 个 Tag)")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, D,
        reprojectionError=5.0, confidence=0.99, iterationsCount=500)

    if not ok:
        print("[错误] solvePnPRansac 失败")
        return

    R, _ = cv2.Rodrigues(rvec)
    n_in = len(inliers) if inliers is not None else 0
    print(f"  RANSAC inliers: {n_in}/{len(obj_pts)}")

    # 重投影误差
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i].reshape(3,1), rvec, tvec, K, D)
        errs.append(np.linalg.norm(proj.flatten() - img_pts[i]))
    avg_err = np.mean(errs)

    # 相机世界位置
    cam_pos = (-R.T @ tvec).flatten()
    print(f"\n结果:")
    print(f"  重投影误差: {avg_err:.2f}px (max={np.max(errs):.2f})")
    print(f"  相机位置: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})m")
    print(f"  相机高度: {abs(cam_pos[2]):.3f}m")

    # 保存 YAML
    yaml_data = {cfg["name"]: {"R": R.tolist(), "t": tvec.flatten().tolist()}}
    yaml_path = f"extrinsics_{cfg['name']}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=None)
    print(f"\n[保存] {yaml_path}")
    print(f"  将此文件内容合并到主项目的 extrinsics.yaml")


# ==============================================================
# 入口
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="相机外参标定")
    parser.add_argument("--camera", choices=["picam", "usb", "usb2"], required=True,
                        help="选择相机")
    parser.add_argument("--mode", choices=["chessboard", "apriltag"],
                        default="apriltag",
                        help="标定模式: chessboard(棋盘格) / apriltag(推荐)")
    parser.add_argument("--intrinsics", default=None,
                        help="内参 JSON 路径 (默认 camera_calibration_{camera}.json)")
    args = parser.parse_args()

    cfg = CAMERA_CONFIGS[args.camera]
    int_path = args.intrinsics or f"camera_calibration_{args.camera}.json"

    if not os.path.exists(int_path):
        print(f"[错误] 未找到内参文件: {int_path}")
        print(f"       请先运行: python calibrate_intrinsics.py --camera {args.camera}")
        sys.exit(1)

    K, D = load_intrinsics(int_path)

    if args.mode == "chessboard":
        calibrate_chessboard(cfg, K, D)
    else:
        calibrate_apriltag(cfg, K, D)

if __name__ == "__main__":
    main()
