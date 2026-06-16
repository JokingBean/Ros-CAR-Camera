#!/usr/bin/env python3
"""
多摄像头AprilTag跟踪器

使用CameraManager统一管理多个摄像头的内参和外参，
支持静态摄像头（手动外参）和动态摄像头（自动标定）。

使用方法：
    # 查看所有摄像头状态
    python multi_camera_tracker.py --status
    
    # 使用指定摄像头运行
    python multi_camera_tracker.py --camera usb_cam_1
    
    # 同时运行多个摄像头
    python multi_camera_tracker.py --multi
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

# 导入摄像头管理器
from camera_manager import CameraManager, CameraConfig, create_camera_capture


# =============================================================================
# 常量定义
# =============================================================================

# 立方体侧面 AprilTag
TARGET_TAG_IDS = [0, 1, 2, 3]
CUBE_TAG_SIZE_M = 0.135

# 立方体尺寸
CUBE_EDGE_SIZE_M = 0.25
FACE_CENTER_TO_CUBE_CENTER_M = CUBE_EDGE_SIZE_M / 2.0

# 侧面 tag 中心距离地面的固定高度
TAG_HEIGHT_FROM_GROUND_M = 0.267

# 实际目标中心相对几何中心向"右侧(tag0)"偏移 4.5mm
CENTER_OFFSET_TO_TAG0_M = 0.0045

# 地面 AprilTag
FLOOR_TAG_SIZE_M = 0.09
MIN_FLOOR_TAGS_FOR_EXTRINSIC = 3


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class TagDetectionPose:
    """单个 AprilTag 检测结果"""
    tvec: np.ndarray
    r_tag_to_cam: np.ndarray
    detect_time_ms: float
    corners: np.ndarray
    center: np.ndarray
    decision_margin: float
    tag_id: int


@dataclass
class TrackedObject:
    """跟踪目标"""
    tag_id: int
    cube_center_cam: np.ndarray
    cube_center_world: np.ndarray
    yaw_deg: float
    confidence: float
    timestamp: float


# =============================================================================
# 检测与标定
# =============================================================================

def build_detector() -> Detector:
    """构造 AprilTag 检测器"""
    return Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tags(
    frame_bgr: np.ndarray,
    detector: Detector,
    k: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float,
    tag_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[TagDetectionPose]]:
    """从单帧 BGR 图像中检测目标 tag"""
    t0 = time.perf_counter()
    
    # 去畸变
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
    """估计 world -> camera 刚体变换"""
    if world_points.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required")
    
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
    """利用地面 tag 的中心点对应关系估计实时外参"""
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
    """将相机坐标系点转换到世界坐标系"""
    return r_wc.T @ (p_cam - t_wc)


def cube_center_camera_from_detection(det_pose: TagDetectionPose) -> np.ndarray:
    """根据立方体侧面 tag，计算正方体中心在相机坐标系中的位置"""
    tag_normal_tag = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    inward_cam = det_pose.r_tag_to_cam @ tag_normal_tag
    inward_cam = inward_cam / max(float(np.linalg.norm(inward_cam)), 1e-9)
    return det_pose.tvec + inward_cam * FACE_CENTER_TO_CUBE_CENTER_M


def face_inward_normal_world_xy(r_wc: np.ndarray, r_tag_to_cam: np.ndarray) -> np.ndarray:
    """计算该 tag 法向量在世界坐标系 xy 平面的单位方向"""
    tag_normal_tag = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal_cam = r_tag_to_cam @ tag_normal_tag
    normal_world = r_wc.T @ normal_cam
    normal_xy = np.array([normal_world[0], normal_world[1]], dtype=np.float64)
    norm = float(np.linalg.norm(normal_xy))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return normal_xy / norm


def rotate_ccw_90(v_xy: np.ndarray) -> np.ndarray:
    """二维向量逆时针旋转 90 度"""
    return np.array([-v_xy[1], v_xy[0]], dtype=np.float64)


def rotate_cw_90(v_xy: np.ndarray) -> np.ndarray:
    """二维向量顺时针旋转 90 度"""
    return np.array([v_xy[1], -v_xy[0]], dtype=np.float64)


def front_xy_from_tag_and_inward(tag_id: int, inward_xy: np.ndarray) -> Optional[np.ndarray]:
    """由 tag 编号和"指向车体内部"的方向，恢复车头方向"""
    if tag_id == 3:
        front_xy = -inward_xy
    elif tag_id == 1:
        front_xy = inward_xy
    elif tag_id == 0:
        front_xy = rotate_ccw_90(inward_xy)
    elif tag_id == 2:
        front_xy = rotate_cw_90(inward_xy)
    else:
        return None
    
    norm = float(np.linalg.norm(front_xy))
    if norm < 1e-9:
        return None
    return front_xy / norm


def body_right_xy_from_front(front_xy: np.ndarray) -> np.ndarray:
    """由车头方向得到车体右侧方向"""
    right_xy = rotate_cw_90(front_xy)
    norm = float(np.linalg.norm(right_xy))
    if norm < 1e-9:
        return np.array([0.0, 0.0], dtype=np.float64)
    return right_xy / norm


def cube_center_world_from_detection(
    r_wc: np.ndarray,
    t_wc: np.ndarray,
    det_pose: TagDetectionPose,
    inward_sign: float = 1.0,
) -> np.ndarray:
    """根据立方体侧面 tag，计算目标中心在世界坐标系中的位置"""
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


def cube_front_direction_world(
    r_wc: np.ndarray,
    det_pose: TagDetectionPose,
    inward_sign: float = 1.0,
) -> Optional[np.ndarray]:
    """根据当前看到的 tag，返回正方体车头方向在世界坐标系中的单位向量"""
    inward_xy = face_inward_normal_world_xy(r_wc, det_pose.r_tag_to_cam) * float(inward_sign)
    front_world_xy = front_xy_from_tag_and_inward(det_pose.tag_id, inward_xy)
    if front_world_xy is None:
        return None
    return np.array([front_world_xy[0], front_world_xy[1], 0.0], dtype=np.float64)


def wrap_deg(angle_deg: float) -> float:
    """将角度归一化到 [0, 360) 区间"""
    return float(angle_deg % 360.0)


def cube_yaw_deg_from_front_world(front_world: np.ndarray) -> float:
    """由世界坐标系中的车头方向向量计算 yaw 角"""
    return wrap_deg(math.degrees(math.atan2(front_world[1], front_world[0])))


# =============================================================================
# 可视化
# =============================================================================

def project_world_point_to_pixel(
    k: np.ndarray,
    r_wc: np.ndarray,
    t_wc: np.ndarray,
    p_world: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """将世界坐标系 3D 点投影到去畸变图像像素坐标"""
    p_cam = r_wc @ p_world + t_wc
    if float(p_cam[2]) <= 1e-6:
        return None
    u = float(k[0, 0]) * float(p_cam[0]) / float(p_cam[2]) + float(k[0, 2])
    v = float(k[1, 1]) * float(p_cam[1]) / float(p_cam[2]) + float(k[1, 2])
    return int(round(u)), int(round(v))


def draw_detection_overlay(
    frame: np.ndarray,
    det_pose: TagDetectionPose,
    cube_center_world: Optional[np.ndarray],
    front_dir_world: Optional[np.ndarray],
    k: np.ndarray,
    r_wc: np.ndarray,
    t_wc: np.ndarray,
) -> np.ndarray:
    """绘制检测结果覆盖"""
    out = frame.copy()
    
    corners_i = det_pose.corners.astype(np.int32)
    center_i = tuple(det_pose.center.astype(np.int32).tolist())
    
    # 绘制 tag 边框
    cv2.polylines(out, [corners_i.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    
    # 绘制中心点
    cv2.circle(out, center_i, 5, (0, 0, 255), -1)
    
    # 绘制 tag ID
    cv2.putText(
        out,
        f"id:{det_pose.tag_id}",
        (center_i[0] + 10, center_i[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    
    # 绘制车头箭头
    if cube_center_world is not None and front_dir_world is not None:
        arrow_start = project_world_point_to_pixel(k, r_wc, t_wc, cube_center_world)
        arrow_tip_world = cube_center_world + front_dir_world * 0.30
        arrow_end = project_world_point_to_pixel(k, r_wc, t_wc, arrow_tip_world)
        
        if arrow_start is not None and arrow_end is not None:
            cv2.arrowedLine(
                out,
                arrow_start,
                arrow_end,
                (255, 0, 255),
                4,
                cv2.LINE_AA,
                tipLength=0.35,
            )
    
    return out


def draw_info_panel(
    frame: np.ndarray,
    tracked_objects: List[TrackedObject],
    camera_name: str,
    extrinsic_status: str,
) -> np.ndarray:
    """绘制信息面板"""
    out = frame.copy()
    
    lines = [
        f"Camera: {camera_name}",
        f"Status: {extrinsic_status}",
        "",
    ]
    
    for obj in tracked_objects:
        lines.append(f"tag{obj.tag_id}: heading={obj.yaw_deg:.1f}deg")
        lines.append(
            f"  world: x={obj.cube_center_world[0]:.2f}, "
            f"y={obj.cube_center_world[1]:.2f}, "
            f"z={obj.cube_center_world[2]:.2f}"
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
    
    x0, y0 = 20, 20
    
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


# =============================================================================
# 单摄像头跟踪器
# =============================================================================

class SingleCameraTracker:
    """单摄像头跟踪器"""
    
    def __init__(
        self,
        camera_config: CameraConfig,
        floor_tag_world_centers: Dict[int, np.ndarray],
        detector: Detector,
    ):
        self.camera_config = camera_config
        self.floor_tag_world_centers = floor_tag_world_centers
        self.detector = detector
        
        # 相机参数
        self.k, self.dist = camera_config.get_intrinsics()
        self.resolution = camera_config.get_resolution()
        
        # 外参状态
        self.r_wc: Optional[np.ndarray] = None
        self.t_wc: Optional[np.ndarray] = None
        self.floor_used_ids: List[int] = []
        
        # 摄像头捕获器
        self.capture = None
    
    def open(self) -> bool:
        """打开摄像头"""
        try:
            self.capture = create_camera_capture(self.camera_config)
            return self.capture.open()
        except Exception as e:
            print(f"[{self.camera_config.name}] Failed to open camera: {e}")
            return False
    
    def close(self) -> None:
        """关闭摄像头"""
        if self.capture:
            self.capture.close()
            self.capture = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TrackedObject]]:
        """处理单帧，返回显示图像和跟踪目标"""
        # 检测立方体标签
        undistorted, cube_detections = detect_tags(
            frame_bgr=frame,
            detector=self.detector,
            k=self.k,
            dist=self.dist,
            tag_size_m=CUBE_TAG_SIZE_M,
            tag_ids=TARGET_TAG_IDS,
        )
        
        # 检测地面标签（仅用于自动标定模式的摄像头）
        _, floor_detections = detect_tags(
            frame_bgr=frame,
            detector=self.detector,
            k=self.k,
            dist=self.dist,
            tag_size_m=FLOOR_TAG_SIZE_M,
            tag_ids=list(self.floor_tag_world_centers.keys()),
        )
        
        # 更新外参（仅自动标定模式）
        if self.camera_config.needs_auto_calibration():
            extrinsic = estimate_extrinsic_from_floor_tags(
                detections=floor_detections,
                floor_tag_world_centers=self.floor_tag_world_centers,
            )
            if extrinsic is not None:
                self.r_wc, self.t_wc, self.floor_used_ids = extrinsic
        
        # 如果是手动模式，使用配置中的外参
        if self.camera_config.is_static():
            self.r_wc, self.t_wc = self.camera_config.get_extrinsics()
        
        # 跟踪目标
        tracked_objects: List[TrackedObject] = []
        display = undistorted.copy()
        
        for det_pose in cube_detections:
            if self.r_wc is not None and self.t_wc is not None:
                # 计算立方体中心世界坐标
                cube_center_world = cube_center_world_from_detection(
                    self.r_wc,
                    self.t_wc,
                    det_pose,
                    inward_sign=1.0,
                )
                
                # 计算车头方向
                front_dir_world = cube_front_direction_world(
                    self.r_wc,
                    det_pose,
                    inward_sign=1.0,
                )
                
                cube_yaw_deg = 0.0
                if front_dir_world is not None:
                    cube_yaw_deg = cube_yaw_deg_from_front_world(front_dir_world)
                
                tracked_objects.append(TrackedObject(
                    tag_id=det_pose.tag_id,
                    cube_center_cam=cube_center_camera_from_detection(det_pose),
                    cube_center_world=cube_center_world,
                    yaw_deg=cube_yaw_deg,
                    confidence=det_pose.decision_margin,
                    timestamp=time.time(),
                ))
                
                # 绘制覆盖
                display = draw_detection_overlay(
                    display,
                    det_pose,
                    cube_center_world,
                    front_dir_world,
                    self.k,
                    self.r_wc,
                    self.t_wc,
                )
        
        # 外参状态文本
        if self.r_wc is not None and self.t_wc is not None:
            if self.camera_config.is_static():
                extrinsic_status = "manual (static)"
            else:
                extrinsic_status = f"auto, floor_tags={len(self.floor_used_ids)}"
        else:
            extrinsic_status = "waiting for calibration..."
        
        # 绘制信息面板
        display = draw_info_panel(
            display,
            tracked_objects,
            self.camera_config.name,
            extrinsic_status,
        )
        
        return display, tracked_objects
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if self.capture is None:
            return None
        return self.capture.read()


# =============================================================================
# 多摄像头管理器
# =============================================================================

class MultiCameraTracker:
    """多摄像头跟踪器"""
    
    def __init__(self, config_path: str = "cameras_config.yaml"):
        # 初始化摄像头管理器
        self.manager = CameraManager(config_path)
        self.detector = build_detector()
        
        # 各摄像头跟踪器
        self.trackers: Dict[str, SingleCameraTracker] = {}
        
        # 初始化各摄像头跟踪器
        for cam_name in self.manager.list_enabled_cameras():
            cam_config = self.manager.get_camera(cam_name)
            if cam_config is None:
                continue
            
            tracker = SingleCameraTracker(
                camera_config=cam_config,
                floor_tag_world_centers=self.manager.floor_tag_world_centers,
                detector=self.detector,
            )
            self.trackers[cam_name] = tracker
    
    def open_all(self) -> Dict[str, bool]:
        """打开所有摄像头，优先打开树莓派摄像头，再打开 USB 摄像头"""
        results = {}
        
        def open_priority(item: Tuple[str, SingleCameraTracker]) -> Tuple[int, str]:
            name, tracker = item
            cam_type = tracker.camera_config.type
            if cam_type == 'picamera':
                return (0, name)
            if cam_type == 'usb':
                return (1, name)
            return (2, name)
        
        for name, tracker in sorted(self.trackers.items(), key=open_priority):
            results[name] = tracker.open()
        return results
    
    def close_all(self) -> None:
        """关闭所有摄像头"""
        for tracker in self.trackers.values():
            tracker.close()
    
    def process_all(self) -> Dict[str, Tuple[np.ndarray, List[TrackedObject]]]:
        """处理所有摄像头的帧"""
        results = {}
        for name, tracker in self.trackers.items():
            frame = tracker.read_frame()
            if frame is not None:
                display, objects = tracker.process_frame(frame)
                results[name] = (display, objects)
        return results


# =============================================================================
# 主程序
# =============================================================================

def run_single_camera(manager: CameraManager, camera_name: str) -> None:
    """运行单个摄像头"""
    cam_config = manager.get_camera(camera_name)
    if cam_config is None:
        print(f"Camera '{camera_name}' not found")
        return
    
    print(f"\nStarting camera: {camera_name}")
    print(f"  Type: {cam_config.type}")
    print(f"  Resolution: {cam_config.resolution}")
    print(f"  Extrinsic mode: {cam_config.extrinsics.mode}")
    
    # 创建跟踪器
    detector = build_detector()
    tracker = SingleCameraTracker(
        camera_config=cam_config,
        floor_tag_world_centers=manager.floor_tag_world_centers,
        detector=detector,
    )
    
    if not tracker.open():
        print(f"Failed to open camera '{camera_name}'")
        return
    
    window_name = f"Camera: {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = tracker.read_frame()
            if frame is None:
                continue
            
            display, objects = tracker.process_frame(frame)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


def run_multi_camera(config_path: str = "cameras_config.yaml") -> None:
    """同时运行多个摄像头"""
    tracker = MultiCameraTracker(config_path)
    
    print("\n[MultiCameraTracker] Initializing cameras...")
    results = tracker.open_all()
    
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    
    if not any(results.values()):
        print("No cameras opened successfully")
        return
    
    # 为每个摄像头创建窗口
    for name in tracker.trackers.keys():
        cv2.namedWindow(f"Camera: {name}", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            results = tracker.process_all()
            
            for name, (display, objects) in results.items():
                cv2.imshow(f"Camera: {name}", display)
            
            # 按 ESC 或 q 退出
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        tracker.close_all()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera AprilTag Tracker")
    parser.add_argument(
        '--config',
        default='cameras_config.yaml',
        help='Camera configuration file'
    )
    parser.add_argument(
        '--camera',
        help='Run specific camera by name'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Run multiple cameras simultaneously'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show camera status'
    )
    
    args = parser.parse_args()
    
    if args.status:
        manager = CameraManager(args.config)
        manager.print_status()
        return
    
    if args.multi:
        run_multi_camera(args.config)
        return
    
    if args.camera:
        manager = CameraManager(args.config)
        run_single_camera(manager, args.camera)
        return
    
    # 默认：显示状态
    manager = CameraManager(args.config)
    manager.print_status()
    print("\nUsage:")
    print("  python multi_camera_tracker.py --status")
    print("  python multi_camera_tracker.py --camera usb_cam_1")
    print("  python multi_camera_tracker.py --multi")


if __name__ == "__main__":
    main()
