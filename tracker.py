"""
多相机 3D 追踪 — ROS-Camera 多相机立方体追踪
=============================================
对每个相机检测到的目标 Tag，使用 solvePnP 求出其相对于该相机的位姿，
再通过相机外参变换到世界坐标系，最后多相机融合。"""

import cv2
import numpy as np


def _tag_object_points(tag_size: float):
    """Tag 自身坐标系下的四个角点（z=0 平面，中心在原点）。"""
    h = tag_size / 2.0
    return np.array([
        [-h, -h, 0],
        [ h, -h, 0],
        [ h,  h, 0],
        [-h,  h, 0],
    ], dtype=np.float64)


def estimate_single_pose(detection, tag_size: float,
                         camera_matrix, dist_coeffs,
                         R_w2c, t_w2c):
    """对单个相机的一个检测，计算该 Tag 在世界坐标系下的位姿。

    参数:
      detection     — pupil-apriltags Detection
      tag_size      — Tag 物理边长（米）
      camera_matrix — 3x3 内参
      dist_coeffs   — 畸变系数
      R_w2c, t_w2c  — 相机外参 (世界→相机)

    返回 dict {tag_id, position, source_camera} 或 None。
    """
    obj_pts = _tag_object_points(tag_size)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, detection.corners, camera_matrix, dist_coeffs)
    if not success:
        return None

    R_t2c, _ = cv2.Rodrigues(rvec)   # Tag → 相机
    t_t2c = tvec.reshape(3, 1)

    # 变换到世界: P_w = R_w2c^T * (P_c - t_w2c)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    R_t2w = R_c2w @ R_t2c
    t_t2w = R_c2w @ t_t2c + t_c2w

    return {
        "tag_id": detection.tag_id,
        "position": t_t2w.flatten(),
        "rotation": R_t2w,
    }


# ======================================================================
class MultiCameraTracker:
    """多相机追踪器 — 聚合各相机检测结果，去重 + 融合。"""

    def __init__(self):
        self._history = {}        # tag_id -> last_position (可用于平滑)

    def update(self, all_results: list, floor_tag_ids: set = None):
        """融合多相机结果。

        参数:
          all_results   — [(cam_name, pose_dict or None), ...]
          floor_tag_ids — 地面 Tag ID 集合（会被跳过）

        返回 [pose_dict, ...]，每个 pose_dict 含
          tag_id, position, confidence, source_cameras
        """
        if floor_tag_ids is None:
            floor_tag_ids = set()

        # 按 tag_id 聚合有效检测
        by_tag = {}
        for cam_name, pose in all_results:
            if pose is None:
                continue
            if pose["tag_id"] in floor_tag_ids:
                continue
            tid = pose["tag_id"]
            pose["_cam"] = cam_name
            by_tag.setdefault(tid, []).append(pose)

        fused = []
        for tid, poses in by_tag.items():
            positions = np.array([p["position"] for p in poses])
            avg_pos = positions.mean(axis=0)

            fused.append({
                "tag_id": tid,
                "position": avg_pos,
                "confidence": min(len(poses) / 3.0, 1.0),
                "source_cameras": [p["_cam"] for p in poses],
            })

        return fused
