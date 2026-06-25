"""
多相机 3D 追踪 — ROS-Camera 多相机立方体追踪
=============================================
对每个相机检测到的 Tag，solvePnP 求位姿 → 世界坐标变换 → GSD 加权融合。

融合策略:
  - GSD 加权: 相机在某点的 GSD 越小，该相机的估计权重越高
  - 置信度评分: Tag 检测的 decision_margin + 重投影误差综合判定
  - 最佳选择: 如只需单个最优结果，返回 GSD 最小的相机估计
"""

import cv2
import numpy as np


def _tag_object_points(tag_size: float):
    h = tag_size / 2.0
    return np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], dtype=np.float64)


def gsd_at_point(x, y, z, K, R_w2c, t_w2c):
    """计算世界坐标某点的 GSD (mm/px)，越小越精细。"""
    P = np.array([[x],[y],[z]], dtype=np.float64)
    P_c = R_w2c @ P + t_w2c
    dist = np.linalg.norm(P_c)
    focal = (K[0,0] + K[1,1]) / 2.0
    return dist / focal * 1000.0


def estimate_single_pose(detection, tag_size: float,
                         camera_matrix, dist_coeffs,
                         R_w2c, t_w2c):
    """单个相机 + 单个检测 → 世界坐标下的 Tag 位姿。

    返回 dict:
      tag_id, position (3,), rotation (3x3),
      gsd (mm/px), reproj_error (px), decision_margin
    """
    obj_pts = _tag_object_points(tag_size)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, detection.corners, camera_matrix, dist_coeffs)
    if not success:
        return None

    R_t2c, _ = cv2.Rodrigues(rvec)
    t_t2c = tvec.reshape(3, 1)

    # 变换到世界
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    R_t2w = R_c2w @ R_t2c
    t_t2w = (R_c2w @ t_t2c + t_c2w).flatten()

    # GSD（用 Tag 中心位置计算）
    gsd = gsd_at_point(t_t2w[0], t_t2w[1], t_t2w[2],
                       camera_matrix, R_w2c, t_w2c)

    # 重投影误差
    proj, _ = cv2.projectPoints(obj_pts.reshape(-1,1,3), rvec, tvec,
                                camera_matrix, dist_coeffs)
    reproj = np.mean([np.linalg.norm(proj[i] - detection.corners[i])
                      for i in range(4)])

    return {
        "tag_id": detection.tag_id,
        "position": t_t2w,
        "rotation": R_t2w,
        "gsd": round(gsd, 2),
        "reproj_error": round(float(reproj), 2),
        "decision_margin": round(float(detection.decision_margin), 1),
    }


# ======================================================================
# 立方体中心计算（方案B：固定偏移，不依赖朝向）
# ======================================================================

# 小车物理参数
CUBE_SIZE = 0.25       # 立方体边长 m
TAG_HEIGHT = 0.267     # Tag 中心离地高度 m
CUBE_CENTER_Z = CUBE_SIZE / 2.0  # 立方体中心 Z = 0.125m

# Tag 面 → 世界 XY 偏移（Tag 位置 → 立方体中心）
# 假设立方体近似轴对齐，Tag 法向指向外侧
# 偏移量 = CUBE_SIZE/2 沿法向的反方向（即指向立方体中心）
TAG_FACE_OFFSET_XY = {
    0: ( CUBE_SIZE/2,  0),               # 左面(x-) → 中心在 +x
    1: ( 0, -CUBE_SIZE/2),               # 前面(y+) → 中心在 -y
    2: (-CUBE_SIZE/2,  0),               # 右面(x+) → 中心在 -x
    3: ( 0,  CUBE_SIZE/2),               # 后面(y-) → 中心在 +y
}


def tag_to_cube_center(tag_id: int, tag_position: np.ndarray):
    """从 Tag 的世界位置推算立方体几何中心（忽略旋转）。

    参数:
      tag_id      — Tag 编号 (0=左, 1=前, 2=右, 3=后)
      tag_position — Tag 在世界坐标系的位置 (3,)

    返回:
      cube_center — (3,) 立方体中心世界坐标
    """
    dx, dy = TAG_FACE_OFFSET_XY.get(tag_id, (0, 0))
    return np.array([
        tag_position[0] + dx,
        tag_position[1] + dy,
        CUBE_CENTER_Z,             # Z 固定，立方体坐地
    ])


# ======================================================================
class MultiCameraTracker:
    """多相机追踪器 — GSD 加权融合 + 最佳选择。"""

    def __init__(self, cam_params: dict = None):
        """
        cam_params: {cam_name: {"K": 3x3, "R": 3x3, "t": (3,1)}}
        """
        self.cam_params = cam_params or {}
        self._history = {}

    # ------------------------------------------------------------------
    def update(self, all_results: list,
               reference_tag_ids: set = None,
               mode: str = "gsd_weighted"):
        """融合多相机结果。

        参数:
          all_results       — [(cam_name, pose_dict or None), ...]
          reference_tag_ids — 参考 Tag（用于相机定位，不追踪）
          mode              — "gsd_weighted" | "best_select" | "average"

        返回:
          [{"tag_id", "position", "confidence", "source_cameras", "best_camera"}, ...]
        """
        if reference_tag_ids is None:
            reference_tag_ids = set()

        # 按 tag_id 聚合
        by_tag = {}
        for cam_name, pose in all_results:
            if pose is None:
                continue
            if pose["tag_id"] in reference_tag_ids:
                continue
            tid = pose["tag_id"]
            pose["_cam"] = cam_name
            by_tag.setdefault(tid, []).append(pose)

        fused = []
        for tid, poses in by_tag.items():
            if mode == "best_select":
                result = self._fuse_best(poses)
            elif mode == "gsd_weighted":
                result = self._fuse_gsd_weighted(poses)
            else:
                result = self._fuse_average(poses)
            fused.append(result)

        return fused

    # ------------------------------------------------------------------
    def _fuse_best(self, poses: list):
        """选择 GSD 最小的相机估计作为最终结果。"""
        best = min(poses, key=lambda p: p["gsd"])
        return {
            "tag_id": best["tag_id"],
            "position": best["position"],
            "confidence": 1.0,
            "source_cameras": [p["_cam"] for p in poses],
            "best_camera": best["_cam"],
            "gsd": best["gsd"],
            "gsd_values": {p["_cam"]: p["gsd"] for p in poses},
        }

    # ------------------------------------------------------------------
    def _fuse_gsd_weighted(self, poses: list):
        """GSD 加权平均：GSD 越小权重越高。weight = 1/GSD，归一化。"""
        weights = np.array([1.0 / max(p["gsd"], 0.01) for p in poses])
        weights /= weights.sum()

        weighted_pos = np.zeros(3)
        for w, p in zip(weights, poses):
            weighted_pos += w * p["position"]

        best = min(poses, key=lambda p: p["gsd"])

        return {
            "tag_id": poses[0]["tag_id"],
            "position": weighted_pos,
            "confidence": round(float(weights.max() / weights.sum()), 3),
            "source_cameras": [p["_cam"] for p in poses],
            "best_camera": best["_cam"],
            "gsd": best["gsd"],
            "weights": {p["_cam"]: round(float(w), 3) for p, w in zip(poses, weights)},
        }

    # ------------------------------------------------------------------
    def _fuse_average(self, poses: list):
        """简单平均（GSD 未知时使用）。"""
        positions = np.array([p["position"] for p in poses])
        avg = positions.mean(axis=0)
        best = min(poses, key=lambda p: p["gsd"])
        return {
            "tag_id": poses[0]["tag_id"],
            "position": avg,
            "confidence": min(len(poses) / 3.0, 1.0),
            "source_cameras": [p["_cam"] for p in poses],
            "best_camera": best["_cam"],
        }
