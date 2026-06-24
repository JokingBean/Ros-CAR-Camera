"""
PnP 外参标定 — ROS-Camera 多相机立方体追踪
===========================================
利用已知世界坐标的地面 AprilTag + solvePnP 求解相机外参 (R, t)。

两种使用场景：
  1) 初次标定 — 系统未知位置，通过地面 Tag 计算相机位姿
  2) 部署重定位 — 相机大致位置已知，用少量 Tag 微调（接口相同）

标定结果 (R, t) 满足：P_cam = R @ P_world + t
"""

import cv2
import numpy as np
import yaml


def _tag_corners_3d(tag_center: np.ndarray, half_size: float):
    """返回 Tag 四个角点的世界坐标（z 不变）。
    Tag 角点顺序与 pupil-apriltags 输出一致（左上→右上→右下→左下）。"""
    x, y, z = tag_center
    return np.array([
        [x - half_size, y - half_size, z],
        [x + half_size, y - half_size, z],
        [x + half_size, y + half_size, z],
        [x - half_size, y + half_size, z],
    ], dtype=np.float64)


def calibrate_extrinsics(detections, floor_tag_map: dict,
                         tag_size: float,
                         camera_matrix, dist_coeffs):
    """利用一帧中检测到的地面 Tag 求解该相机的外参。

    参数:
      detections    — pupil-apriltags Detection 列表（当前帧）
      floor_tag_map — {tag_id: (x, y, z)}, 地面 Tag 世界坐标字典
      tag_size      — 地面 Tag 物理边长（米）
      camera_matrix — 3x3 内参矩阵
      dist_coeffs   — 畸变系数 (1x5 或 5x1)

    返回:
      (R, t) 成功；None 失败
        R — 3x3 旋转矩阵 (世界→相机)
        t — 3x1 平移向量 (世界→相机)
        满足 P_cam = R @ P_world + t
    """
    obj_pts, img_pts = [], []
    half = tag_size / 2.0

    for det in detections:
        tid = det.tag_id
        if tid not in floor_tag_map:
            continue
        center_3d = floor_tag_map[tid]                     # (3,) tuple
        corners_3d = _tag_corners_3d(np.array(center_3d), half)
        for c3, c2 in zip(corners_3d, det.corners):
            obj_pts.append(c3)
            img_pts.append(c2)

    if len(obj_pts) < 4:
        print("[标定] 地面 Tag 点数不足 (<4)")
        return None

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                       camera_matrix, dist_coeffs)
    if not success:
        print("[标定] solvePnP 失败")
        return None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def compute_reprojection_error(detections, floor_tag_map: dict,
                               tag_size: float,
                               R, t, camera_matrix, dist_coeffs):
    """计算重投影误差（像素），评估外参质量。"""
    half = tag_size / 2.0
    errors = []

    for det in detections:
        tid = det.tag_id
        if tid not in floor_tag_map:
            continue
        center_3d = floor_tag_map[tid]
        corners_3d = _tag_corners_3d(np.array(center_3d), half)

        # 世界 → 相机 → 图像
        cam_pts = (R @ corners_3d.T + t).T              # (4, 3)
        proj, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3),
                                    camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)

        for p_proj, p_det in zip(proj, det.corners):
            errors.append(np.linalg.norm(p_proj - p_det))

    return np.mean(errors) if errors else 999.0


# ======================================================================
# 外参持久化
# ======================================================================

def save_extrinsics(extrinsics: dict, path: str):
    """保存 {cam_name: {R: [[...]], t: [...]}} 到 YAML。"""
    serializable = {}
    for cam_name, (R, t) in extrinsics.items():
        serializable[cam_name] = {
            "R": R.tolist(),
            "t": t.flatten().tolist(),
        }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(serializable, f, default_flow_style=None)
    print(f"[标定] 外参已保存至 {path}")


def load_extrinsics(path: str):
    """从 YAML 加载 {cam_name: (R, t)}，文件不存在返回 None。"""
    import os
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    extrinsics = {}
    for cam_name, data in raw.items():
        R = np.array(data["R"], dtype=np.float64)
        t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        extrinsics[cam_name] = (R, t)
    print(f"[标定] 已加载外参 {path} ({len(extrinsics)} 相机)")
    return extrinsics
