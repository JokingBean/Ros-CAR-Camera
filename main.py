"""
ROS-Camera 多相机立方体追踪 — 主程序入口
========================================
使用流程：
  首次运行 : python main.py                 # 自动进入标定模式
  日常运行 : python main.py                 # 加载已有外参
  强制重标定: python main.py --force-calib
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml
import numpy as np

from camera_reader import open_all_cameras, close_all_cameras, CameraReader
from detector import TagDetector
from calibrator import (calibrate_extrinsics, compute_reprojection_error,
                        save_extrinsics, load_extrinsics)
from tracker import estimate_single_pose, MultiCameraTracker

# fmt: off
# ======================================================================
# 工具
# ======================================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_floor_tags(path="floor_tags.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
            for k, v in raw["tags"].items()}


def _cam_params(cfg: dict):
    """从单相机配置中提取 (camera_matrix, dist_coeffs)。"""
    K = cfg["camera_matrix"]
    cam_mat = np.array([[K["fx"], 0, K["cx"]],
                        [0, K["fy"], K["cy"]],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.array(cfg["dist_coeffs"], dtype=np.float64)
    return cam_mat, dist


# ======================================================================
# 标定模式
# ======================================================================

def run_calibration(cameras: list[CameraReader],
                    detector: TagDetector,
                    config: dict, floor_tags: dict):
    """交互式标定：用户按 'c' 触发，每台相机独立标定。"""
    print("\n===== 标定模式 =====")
    print("按 'c' 采集并标定 | 按 'q' 退出\n")

    calib_cfg = config["calibration"]
    tag_size = config["floor_tag_size"]
    extrinsics = {}
    calibrated = set()

    # 使用第一台相机的窗口来控制
    windows = []
    for i, cam in enumerate(cameras):
        win = f"calib_{cam.name}"
        windows.append(win)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)

    while True:
        # 读帧
        frames = {}
        for cam in cameras:
            frames[cam.name] = cam.read()

        # 检测
        cam_detections = {}
        for cam in cameras:
            f = frames.get(cam.name)
            if f is not None:
                cam_detections[cam.name] = detector.detect(f)

        # 显示
        for cam in cameras:
            f = frames.get(cam.name)
            if f is None:
                continue
            dets = cam_detections.get(cam.name, [])
            annotated = f.copy()
            for d in dets:
                pts = d.corners.astype(int)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cx, cy = pts.mean(axis=0)
                color = (0, 255, 0) if cam.name in calibrated else (255, 255, 0)
                cv2.putText(annotated, f"ID:{d.tag_id}", (cx, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status = "OK" if cam.name in calibrated else "WAIT"
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if cam.name in calibrated else (0, 255, 255), 2)
            cv2.imshow(f"calib_{cam.name}", annotated)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            for cam in cameras:
                dets = cam_detections.get(cam.name, [])
                floor_dets = [d for d in dets if d.tag_id in floor_tags]
                if len(floor_dets) < calib_cfg["min_floor_tags"]:
                    print(f"[标定] {cam.name} 地面Tag不足 "
                          f"({len(floor_dets)}/{calib_cfg['min_floor_tags']}), 跳过")
                    continue

                K, dist = _cam_params(
                    next(c for c in config["cameras"] if c["name"] == cam.name))
                result = calibrate_extrinsics(
                    floor_dets, floor_tags, tag_size, K, dist)
                if result is None:
                    print(f"[标定] {cam.name} 标定失败")
                    continue

                R, t = result
                err = compute_reprojection_error(
                    floor_dets, floor_tags, tag_size, R, t, K, dist)
                print(f"[标定] {cam.name} 标定完成  重投影误差: {err:.2f} px")

                if err < calib_cfg["max_repro_error"]:
                    extrinsics[cam.name] = (R, t)
                    calibrated.add(cam.name)
                else:
                    print(f"[标定] {cam.name} 误差超标 ({err:.2f} > "
                          f"{calib_cfg['max_repro_error']}), 重试")

        if len(calibrated) == len(cameras):
            print("\n[标定] 全部相机标定完成!")
            break

    for w in windows:
        cv2.destroyWindow(w)

    if len(extrinsics) == len(cameras):
        save_extrinsics(extrinsics, config["runtime"]["extrinsics_file"])
        return extrinsics

    print("[标定] 未完成全部标定，退出")
    sys.exit(1)


# ======================================================================
# 追踪主循环
# ======================================================================

def run_tracking(cameras: list[CameraReader],
                 detector: TagDetector,
                 config: dict, floor_tags: dict,
                 extrinsics: dict):
    """追踪主循环。"""
    tag_size = config["target_tag_size"]
    floor_ids = set(floor_tags.keys())
    tracker = MultiCameraTracker()

    # 预计算每台相机的内参 + 外参
    cam_params = {}
    for cam in cameras:
        cfg = next(c for c in config["cameras"] if c["name"] == cam.name)
        K, dist = _cam_params(cfg)
        R, t = extrinsics[cam.name]
        cam_params[cam.name] = (K, dist, R, t)

    target_fps = config["runtime"]["target_fps"]
    interval = 1.0 / target_fps

    # 窗口
    for cam in cameras:
        cv2.namedWindow(cam.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam.name, 640, 480)

    print("\n===== 开始追踪 =====")
    print("按 'q' 退出\n")

    while True:
        t0 = cv2.getTickCount()

        # 1) 读帧
        frames = {}
        for cam in cameras:
            frames[cam.name] = cam.read()

        # 2) 检测
        all_results = []
        for cam in cameras:
            f = frames.get(cam.name)
            if f is None:
                continue
            dets = detector.detect(f)
            K, dist, R, t = cam_params[cam.name]
            annotated = f.copy()

            for d in dets:
                pts = d.corners.astype(int)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(annotated, f"ID:{d.tag_id}", (cx, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 跳过地面 Tag，只追踪目标 Tag
                if d.tag_id in floor_ids:
                    continue

                pose = estimate_single_pose(d, tag_size, K, dist, R, t)
                if pose is not None:
                    pos = pose["position"]
                    cv2.putText(annotated, f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})",
                                (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 0), 1)
                all_results.append((cam.name, pose))

            cv2.imshow(cam.name, annotated)

        # 3) 融合
        tracked = tracker.update(all_results, floor_ids)

        # 4) 输出
        for obj in tracked:
            pos = obj["position"]
            cams = obj["source_cameras"]
            print(f"[追踪] ID:{obj['tag_id']:3d}  "
                  f"位置:({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                  f"置信度:{obj['confidence']:.2f}  "
                  f"来源:{','.join(cams)}")

        # 5) 帧率控制
        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        if elapsed < interval:
            cv2.waitKey(max(1, int((interval - elapsed) * 1000)))
        else:
            cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        cv2.destroyWindow(cam.name)


# ======================================================================
# 入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="ROS-Camera 多相机立方体追踪")
    parser.add_argument("--force-calib", action="store_true",
                        help="强制重新标定")
    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    floor_tags = load_floor_tags()
    print(f"[启动] 地面 Tag: {len(floor_tags)} 个")

    # 打开相机
    cameras = open_all_cameras(config)
    if not cameras:
        print("[错误] 没有可用的相机")
        sys.exit(1)
    print(f"[启动] {len(cameras)} 台相机已就绪")

    detector = TagDetector()

    # 尝试加载外参
    ext_path = config["runtime"]["extrinsics_file"]
    extrinsics = load_extrinsics(ext_path) if not args.force_calib else None

    # 需要标定？
    if extrinsics is None:
        extrinsics = run_calibration(cameras, detector, config, floor_tags)
    else:
        # 快速校验：采集一帧检查重投影误差
        print("[校验] 采集帧进行外参校验...")
        import time
        time.sleep(0.5)  # 等相机稳定

        all_ok = True
        for cam in cameras:
            frame = cam.read()
            if frame is None:
                continue
            dets = detector.detect(frame)
            floor_dets = [d for d in dets if d.tag_id in floor_tags]
            if len(floor_dets) < 3:
                print(f"[校验] {cam.name} 地面 Tag 过少，跳过校验")
                continue

            K, dist = _cam_params(
                next(c for c in config["cameras"] if c["name"] == cam.name))
            R, t = extrinsics[cam.name]
            err = compute_reprojection_error(
                floor_dets, floor_tags, config["floor_tag_size"],
                R, t, K, dist)
            print(f"[校验] {cam.name} 重投影误差: {err:.2f} px")
            if err > config["calibration"]["max_repro_error"]:
                print(f"[校验] {cam.name} 误差超标，触发重标定")
                all_ok = False

        if not all_ok:
            print("[校验] 外参校验未通过，进入标定模式")
            extrinsics = run_calibration(cameras, detector, config, floor_tags)

    # 进入追踪
    try:
        run_tracking(cameras, detector, config, floor_tags, extrinsics)
    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    finally:
        close_all_cameras(cameras)
        cv2.destroyAllWindows()
        print("[退出] 程序结束")


if __name__ == "__main__":
    main()
