#!/usr/bin/env python3
"""Real-time AprilTag cube center localization using floor tags for live extrinsic calibration.

功能说明：
1) 打开固定摄像头实时预览。
2) 使用地面 AprilTag（位置来自 floor_tag_layout.yaml，边长 9cm）实时估计相机外参。
3) 使用立方体侧面 AprilTag（tag 0/1/2/3，边长 13.5cm）估计正方体中心。
4) 在画面中显示：
   - 立方体 tag 边框/中心点/ID
   - 正方体中心相机坐标
   - 正方体中心世界坐标（当检测到足够地面 tag 时）
   - 当前用于外参估计的地面 tag 数量

说明：
- 已完全删除离线标定流程。
- 不再使用 ground_truth_points.yaml、extrinsic_snapshots、results 输出。
- 外参仅由实时画面中的地面 tag 动态估计。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

# =========================
# Macros
# =========================
CAMERA_DEVICE_INDEX = 0

# 立方体侧面 AprilTag
TARGET_TAG_IDS = [0, 1, 2, 3]
CUBE_TAG_SIZE_M = 0.135

# 立方体尺寸
CUBE_EDGE_SIZE_M = 0.25
FACE_CENTER_TO_CUBE_CENTER_M = CUBE_EDGE_SIZE_M / 2.0

# 侧面 tag 中心距离地面的固定高度
TAG_HEIGHT_FROM_GROUND_M = 0.267

# 实际目标中心相对几何中心向“右侧(tag0)”偏移 4.5mm
CENTER_OFFSET_TO_TAG0_M = 0.0045

# 地面 AprilTag
FLOOR_TAG_SIZE_M = 0.09
FLOOR_TAG_CONFIG_PATH = "floor_tag_layout.yaml"
MIN_FLOOR_TAGS_FOR_EXTRINSIC = 3

# 立方体 tag 与地面 tag 不能共用同一组 id。
# floor_tag_layout.yaml 中若包含 0/1/2/3，会自动从地面外参标定集合中剔除。


@dataclass
class CameraParams:
    """相机参数。"""

    k: np.ndarray
    dist: np.ndarray
    resolution: Tuple[int, int]
    fps: int


@dataclass
class TagDetectionPose:
    """单个 AprilTag 检测结果。"""

    tvec: np.ndarray
    r_tag_to_cam: np.ndarray
    detect_time_ms: float
    corners: np.ndarray
    center: np.ndarray
    decision_margin: float
    tag_id: int


def load_camera_params(camera_yaml_path: Path, camera_name: str) -> CameraParams:
    """从 camera_intrinsics.yaml 读取内参与分辨率/fps。"""
    with camera_yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg["cameras"][camera_name]
    k = np.array(cam_cfg["camera_matrix"], dtype=np.float64)
    dist = np.array(cam_cfg["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    resolution_cfg = cam_cfg.get("resolution", [1280, 720])
    resolution = (int(resolution_cfg[0]), int(resolution_cfg[1]))
    fps = int(cam_cfg.get("fps", 30))

    return CameraParams(
        k=k,
        dist=dist,
        resolution=resolution,
        fps=fps,
    )


def load_floor_tag_config(tag_config_path: Path) -> Dict[int, np.ndarray]:
    """读取地面 tag 配置，返回 tag_id -> 世界坐标中心。

    注意：
    - 立方体自身使用 tag 0/1/2/3
    - 地面 tag 若也配置了这些 id，会与目标物冲突
    - 因此这里会自动剔除 TARGET_TAG_IDS
    """
    with tag_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("floor_tag_layout.yaml must be a mapping from tag id to xyz list.")

    out: Dict[int, np.ndarray] = {}
    cube_tag_ids = set(TARGET_TAG_IDS)

    for key, value in cfg.items():
        tag_id = int(key)
        if tag_id in cube_tag_ids:
            continue
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"Invalid tag config for id={key}: {value}")
        out[tag_id] = np.array(value, dtype=np.float64).reshape(3)
    return out


def build_detector() -> Detector:
    """构造 AprilTag 检测器。"""
    return Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tag_poses_from_bgr(
    frame_bgr: np.ndarray,
    detector: Detector,
    k: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float,
    tag_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[TagDetectionPose]]:
    """从单帧 BGR 图像中检测目标 tag。"""
    t0 = time.perf_counter()

    undistorted = cv2.undistort(frame_bgr, k, dist)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size_m,
    )

    target_ids = None if tag_ids is None else {int(v) for v in tag_ids}
    detect_time_ms = float((time.perf_counter() - t0) * 1000.0)

    out: List[TagDetectionPose] = []
    for det in detections:
        if target_ids is not None and int(det.tag_id) not in target_ids:
            continue
        out.append(
            TagDetectionPose(
                tvec=np.array(det.pose_t, dtype=np.float64).reshape(3),
                r_tag_to_cam=np.array(det.pose_R, dtype=np.float64).reshape(3, 3),
                detect_time_ms=detect_time_ms,
                corners=np.array(det.corners, dtype=np.float64).reshape(4, 2),
                center=np.array(det.center, dtype=np.float64).reshape(2),
                decision_margin=float(det.decision_margin),
                tag_id=int(det.tag_id),
            )
        )

    out.sort(key=lambda d: float(d.decision_margin), reverse=True)
    return undistorted, out


def estimate_world_to_camera_transform(
    world_points: np.ndarray,
    camera_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """估计 world -> camera 刚体变换，满足 p_cam = R * p_world + t。"""
    if world_points.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required.")

    centroid_a = np.mean(world_points, axis=0)
    centroid_b = np.mean(camera_points, axis=0)

    a_centered = world_points - centroid_a
    b_centered = camera_points - centroid_b

    h = a_centered.T @ b_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = centroid_b - r @ centroid_a
    return r, t


def estimate_extrinsic_from_floor_tags(
    detections: List[TagDetectionPose],
    floor_tag_world_centers: Dict[int, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """利用地面 tag 的中心点对应关系估计实时外参。"""
    world_pts: List[np.ndarray] = []
    cam_pts: List[np.ndarray] = []
    used_ids: List[int] = []

    for det in detections:
        if det.tag_id not in floor_tag_world_centers:
            continue
        world_pts.append(floor_tag_world_centers[det.tag_id])
        cam_pts.append(det.tvec)
        used_ids.append(det.tag_id)

    if len(world_pts) < MIN_FLOOR_TAGS_FOR_EXTRINSIC:
        return None

    r_wc, t_wc = estimate_world_to_camera_transform(
        world_points=np.vstack(world_pts),
        camera_points=np.vstack(cam_pts),
    )
    return r_wc, t_wc, used_ids


def to_world_point(r_wc: np.ndarray, t_wc: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    """将相机坐标系点转换到世界坐标系。"""
    return r_wc.T @ (p_cam - t_wc)


def cube_center_camera_from_detection(det_pose: TagDetectionPose) -> np.ndarray:
    """根据立方体侧面 tag，计算正方体中心在相机坐标系中的位置。

    这里采用 pupil_apriltags 返回的 tag 局部 z 轴方向作为“从 tag 面中心指向立方体内部”的方向。
    现场验证中如果使用减号，得到的结果会更像 tag 面中心；改为加号后与正方体中心几何关系一致。
    """
    tag_normal_tag = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    inward_cam = det_pose.r_tag_to_cam @ tag_normal_tag
    inward_cam = inward_cam / max(float(np.linalg.norm(inward_cam)), 1e-9)
    return det_pose.tvec + inward_cam * FACE_CENTER_TO_CUBE_CENTER_M


def face_inward_normal_world_xy(r_wc: np.ndarray, r_tag_to_cam: np.ndarray) -> np.ndarray:
    """计算该 tag 法向量在世界坐标系 xy 平面的单位方向。符号后续再做一致性判定。"""
    tag_normal_tag = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal_cam = r_tag_to_cam @ tag_normal_tag
    normal_world = r_wc.T @ normal_cam
    normal_xy = np.array([normal_world[0], normal_world[1]], dtype=np.float64)
    norm = float(np.linalg.norm(normal_xy))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return normal_xy / norm


def front_xy_from_tag_and_inward(tag_id: int, inward_xy: np.ndarray) -> Optional[np.ndarray]:
    """由 tag 编号和“指向车体内部”的方向，恢复车头方向。

    新约定：
    - tag3: front
    - tag1: back
    - tag0: right
    - tag2: left
    """
    if tag_id == 3:
        front_xy = -inward_xy
    elif tag_id == 1:
        front_xy = inward_xy
    elif tag_id == 0:
        # right face: inward 指向车体左侧，因此车头 = inward 逆时针转 90°
        front_xy = rotate_ccw_90(inward_xy)
    elif tag_id == 2:
        # left face: inward 指向车体右侧，因此车头 = inward 顺时针转 90°
        front_xy = rotate_cw_90(inward_xy)
    else:
        return None

    norm = float(np.linalg.norm(front_xy))
    if norm < 1e-9:
        return None
    return front_xy / norm


def cube_center_world_from_detection(
    r_wc: np.ndarray,
    t_wc: np.ndarray,
    det_pose: TagDetectionPose,
    inward_sign: float = 1.0,
) -> np.ndarray:
    """根据立方体侧面 tag，计算目标中心在世界坐标系中的位置。"""
    tag_center_world = to_world_point(r_wc, t_wc, det_pose.tvec)
    tag_center_world[2] = TAG_HEIGHT_FROM_GROUND_M

    inward_xy = face_inward_normal_world_xy(r_wc, det_pose.r_tag_to_cam) * float(inward_sign)

    cube_center_world = tag_center_world.copy()
    cube_center_world[0] += inward_xy[0] * FACE_CENTER_TO_CUBE_CENTER_M
    cube_center_world[1] += inward_xy[1] * FACE_CENTER_TO_CUBE_CENTER_M

    # 整体中心向右侧(tag0)偏移 4.5mm
    front_world_xy = front_xy_from_tag_and_inward(det_pose.tag_id, inward_xy)
    if front_world_xy is not None:
        right_world_xy = body_right_xy_from_front(front_world_xy)
        cube_center_world[0] += right_world_xy[0] * CENTER_OFFSET_TO_TAG0_M
        cube_center_world[1] += right_world_xy[1] * CENTER_OFFSET_TO_TAG0_M

    cube_center_world[2] = TAG_HEIGHT_FROM_GROUND_M
    return cube_center_world


def wrap_deg(angle_deg: float) -> float:
    """将角度归一化到 [0, 360) 区间。"""
    return float(angle_deg % 360.0)


def rotate_ccw_90(v_xy: np.ndarray) -> np.ndarray:
    """二维向量逆时针旋转 90 度。"""
    return np.array([-v_xy[1], v_xy[0]], dtype=np.float64)


def rotate_cw_90(v_xy: np.ndarray) -> np.ndarray:
    """二维向量顺时针旋转 90 度。"""
    return np.array([v_xy[1], -v_xy[0]], dtype=np.float64)


def body_right_xy_from_front(front_xy: np.ndarray) -> np.ndarray:
    """由车头方向得到车体右侧方向。"""
    right_xy = rotate_cw_90(front_xy)
    norm = float(np.linalg.norm(right_xy))
    if norm < 1e-9:
        return np.array([0.0, 0.0], dtype=np.float64)
    return right_xy / norm


def cube_front_direction_camera(det_pose: TagDetectionPose) -> Optional[np.ndarray]:
    """根据当前看到的 tag，返回正方体正面方向在相机坐标系中的单位向量。"""
    tag_normal_tag = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    inward_cam = det_pose.r_tag_to_cam @ tag_normal_tag
    inward_cam = inward_cam / max(float(np.linalg.norm(inward_cam)), 1e-9)

    if det_pose.tag_id == 3:
        front_cam = -inward_cam
    elif det_pose.tag_id == 1:
        front_cam = inward_cam
    elif det_pose.tag_id == 0:
        front_xy = rotate_ccw_90(inward_cam[:2])
        front_cam = np.array([front_xy[0], front_xy[1], 0.0], dtype=np.float64)
    elif det_pose.tag_id == 2:
        front_xy = rotate_cw_90(inward_cam[:2])
        front_cam = np.array([front_xy[0], front_xy[1], 0.0], dtype=np.float64)
    else:
        return None

    norm = float(np.linalg.norm(front_cam))
    if norm < 1e-9:
        return None
    return front_cam / norm


def cube_front_direction_world(
    r_wc: np.ndarray,
    det_pose: TagDetectionPose,
    inward_sign: float = 1.0,
) -> Optional[np.ndarray]:
    """根据当前看到的 tag，返回正方体车头方向在世界坐标系中的单位向量。"""
    inward_xy = face_inward_normal_world_xy(r_wc, det_pose.r_tag_to_cam) * float(inward_sign)
    front_world_xy = front_xy_from_tag_and_inward(det_pose.tag_id, inward_xy)
    if front_world_xy is None:
        return None
    return np.array([front_world_xy[0], front_world_xy[1], 0.0], dtype=np.float64)


def cube_yaw_deg_from_front_world(front_world: np.ndarray) -> float:
    """由世界坐标系中的车头方向向量计算 yaw 角。"""
    return wrap_deg(math.degrees(math.atan2(front_world[1], front_world[0])))


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """返回两个角度的最小差值（度）。"""
    d = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return float(d)




def project_camera_point_to_pixel(k: np.ndarray, p_cam: np.ndarray) -> Optional[Tuple[int, int]]:
    """将相机坐标系 3D 点投影到去畸变图像像素坐标。"""
    if float(p_cam[2]) <= 1e-6:
        return None

    u = float(k[0, 0]) * float(p_cam[0]) / float(p_cam[2]) + float(k[0, 2])
    v = float(k[1, 1]) * float(p_cam[1]) / float(p_cam[2]) + float(k[1, 2])
    return int(round(u)), int(round(v))


def project_world_point_to_pixel(
    k: np.ndarray,
    r_wc: np.ndarray,
    t_wc: np.ndarray,
    p_world: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """将世界坐标系 3D 点投影到去畸变图像像素坐标。"""
    p_cam = r_wc @ p_world + t_wc
    return project_camera_point_to_pixel(k, p_cam)


def draw_detection_overlay(
    frame: np.ndarray,
    det_pose: TagDetectionPose,
    arrow_start_px: Optional[Tuple[int, int]] = None,
    arrow_end_px: Optional[Tuple[int, int]] = None,
    arrow_color: Tuple[int, int, int] = (0, 165, 255),
    arrow_thickness: int = 4,
    id_text: Optional[str] = None,
) -> np.ndarray:
    """只在目标附近绘制框、中心点、角点编号和车头箭头。"""
    out = frame.copy()

    corners_i = det_pose.corners.astype(np.int32)
    center_i = tuple(det_pose.center.astype(np.int32).tolist())

    cv2.polylines(out, [corners_i.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cv2.circle(out, center_i, 5, (0, 0, 255), -1)

    for idx, pt in enumerate(corners_i):
        cv2.circle(out, tuple(pt.tolist()), 4, (255, 0, 0), -1)
        cv2.putText(
            out,
            str(idx),
            (int(pt[0]) + 6, int(pt[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        id_text if id_text is not None else f"id:{det_pose.tag_id}",
        (center_i[0] + 10, center_i[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if arrow_start_px is not None and arrow_end_px is not None:
        cv2.arrowedLine(
            out,
            arrow_start_px,
            arrow_end_px,
            arrow_color,
            arrow_thickness,
            cv2.LINE_AA,
            tipLength=0.35,
        )

    return out


def draw_info_panel(
    frame: np.ndarray,
    per_tag_infos: List[Tuple[int, np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """在画面左上角绘制每个 tag 的独立结果，不再显示 fused。"""
    out = frame.copy()

    lines: List[str] = []
    for tag_id, tag_center_world, cube_center_world, cube_yaw_deg in per_tag_infos:
        lines.append(f"tag{tag_id}: heading={cube_yaw_deg:.1f} deg")
        lines.append(
            f"  tag_world:  x={tag_center_world[0]:.2f}, y={tag_center_world[1]:.2f}, z={tag_center_world[2]:.2f}"
        )
        lines.append(
            f"  cube_world: x={cube_center_world[0]:.2f}, y={cube_center_world[1]:.2f}, z={cube_center_world[2]:.2f}"
        )

    if not lines:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_height = 26
    padding = 12

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_w = max(size[0] for size in sizes)
    box_w = max_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x0, y0 = 20, 115

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0.0)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 255), 2)

    for i, line in enumerate(lines):
        y = y0 + padding + 20 + i * line_height
        cv2.putText(
            out,
            line,
            (x0 + padding, y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out


def run_live_camera(
    cam: CameraParams,
    detector: Detector,
    floor_tag_world_centers: Dict[int, np.ndarray],
) -> None:
    """实时打开摄像头，利用地面 tag 动态标定外参并定位立方体中心。"""
    device_index = CAMERA_DEVICE_INDEX

    # 某些 Windows 摄像头在 CAP_DSHOW 下会成功打开但只返回黑帧。
    # 因此优先尝试默认后端，再回退到 DSHOW。
    cap = cv2.VideoCapture(device_index)
    backend_name = "default"

    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        backend_name = "CAP_DSHOW"

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device_index={device_index}")

    print(f"[INFO] camera opened: device_index={device_index}, backend={backend_name}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cam.resolution[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cam.resolution[1]))
    cap.set(cv2.CAP_PROP_FPS, float(cam.fps))

    window_name = "AprilTag Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from camera.")

            # 立方体 tag 单独按 13.5cm 检测
            undistorted, cube_detections = detect_tag_poses_from_bgr(
                frame_bgr=frame,
                detector=detector,
                k=cam.k,
                dist=cam.dist,
                tag_size_m=CUBE_TAG_SIZE_M,
                tag_ids=TARGET_TAG_IDS,
            )

            # 地面 tag 单独按 9cm 检测，用于实时估计外参
            _, floor_detections = detect_tag_poses_from_bgr(
                frame_bgr=frame,
                detector=detector,
                k=cam.k,
                dist=cam.dist,
                tag_size_m=FLOOR_TAG_SIZE_M,
                tag_ids=list(floor_tag_world_centers.keys()),
            )

            live_r_wc: Optional[np.ndarray] = None
            live_t_wc: Optional[np.ndarray] = None
            floor_used_ids: List[int] = []

            extrinsic = estimate_extrinsic_from_floor_tags(
                detections=floor_detections,
                floor_tag_world_centers=floor_tag_world_centers,
            )
            if extrinsic is not None:
                live_r_wc, live_t_wc, floor_used_ids = extrinsic

            display = undistorted.copy()

            per_tag_infos: List[Tuple[int, np.ndarray, np.ndarray, float]] = []

            for det_pose in cube_detections:
                per_arrow_start_px = None
                per_arrow_end_px = None

                if live_r_wc is not None and live_t_wc is not None:
                    per_tag_center_world = to_world_point(live_r_wc, live_t_wc, det_pose.tvec)
                    per_tag_center_world[2] = TAG_HEIGHT_FROM_GROUND_M
                    per_cube_center_world = cube_center_world_from_detection(
                        live_r_wc,
                        live_t_wc,
                        det_pose,
                        inward_sign=1.0,
                    )
                    per_front_dir_world = cube_front_direction_world(
                        live_r_wc,
                        det_pose,
                        inward_sign=1.0,
                    )
                    if per_front_dir_world is not None:
                        per_cube_yaw_deg = cube_yaw_deg_from_front_world(per_front_dir_world)
                        per_tag_infos.append(
                            (
                                det_pose.tag_id,
                                per_tag_center_world,
                                per_cube_center_world,
                                per_cube_yaw_deg,
                            )
                        )
                        per_arrow_start_px = project_world_point_to_pixel(
                            cam.k, live_r_wc, live_t_wc, per_cube_center_world
                        )
                        per_arrow_tip_world = per_cube_center_world + per_front_dir_world * 0.30
                        per_arrow_end_px = project_world_point_to_pixel(
                            cam.k, live_r_wc, live_t_wc, per_arrow_tip_world
                        )

                display = draw_detection_overlay(
                    display,
                    det_pose,
                    arrow_start_px=per_arrow_start_px,
                    arrow_end_px=per_arrow_end_px,
                    arrow_color=(255, 0, 255),
                    arrow_thickness=4,
                    id_text=f"id:{det_pose.tag_id}",
                )

            if per_tag_infos:
                display = draw_info_panel(
                    display,
                    per_tag_infos,
                )

            now = time.perf_counter()
            preview_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            status_text = (
                f"camera={device_index}  "
                f"resolution={cam.resolution[0]}x{cam.resolution[1]}  "
                f"fps={preview_fps:.1f}  "
                f"cube_tags={TARGET_TAG_IDS}  "
                f"cube_detected={len(cube_detections)}  "
                f"floor_tags_for_extrinsic={len(floor_used_ids)}"
            )
            cv2.putText(
                display,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if cube_detections else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Press q or ESC to quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if live_r_wc is not None and live_t_wc is not None:
                cv2.putText(
                    display,
                    f"live extrinsic: OK, floor tag ids={floor_used_ids[:8]}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    f"live extrinsic: need >= {MIN_FLOOR_TAGS_FOR_EXTRINSIC} floor tags",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Real-time AprilTag cube localization using floor tags only"
    )
    parser.add_argument(
        "--camera-yaml",
        default="camera_intrinsics.yaml",
        help="Path to camera_intrinsics.yaml",
    )
    parser.add_argument("--camera-name", default="cam_1", help="Camera key under cameras")
    return parser


def main() -> None:
    """主入口。仅保留实时模式。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_yaml = Path(args.camera_yaml)
    if not camera_yaml.exists():
        raise FileNotFoundError(f"camera intrinsics yaml not found: {camera_yaml}")

    floor_tag_config_path = Path(FLOOR_TAG_CONFIG_PATH)
    if not floor_tag_config_path.exists():
        raise FileNotFoundError(f"floor tag layout not found: {floor_tag_config_path}")

    cam = load_camera_params(camera_yaml, args.camera_name)
    floor_tag_world_centers = load_floor_tag_config(floor_tag_config_path)
    print(
        f"[INFO] floor tag layout loaded: {len(floor_tag_world_centers)} tags "
        f"(excluded cube tag ids={TARGET_TAG_IDS})"
    )
    detector = build_detector()

    run_live_camera(
        cam=cam,
        detector=detector,
        floor_tag_world_centers=floor_tag_world_centers,
    )


if __name__ == "__main__":
    main()
