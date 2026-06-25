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


# 车头朝向（在 Tag 本地坐标系中，车头+0Y 的方向向量）
# Tag 贴在立方体侧面，车头朝向定义为面1→面3方向
# 面1(前): Tag外法向=车头朝向  面2(右): 车头在Tag左侧
# 面3(后): Tag外法向反=车头   面0(左): 车头在Tag右侧
_HEADING_IN_TAG_FRAME = {
    0: np.array([1, 0, 0]),    # 左面: 车头指向 Tag 的 +X
    1: np.array([0, 0, 1]),    # 前面: 车头 = Tag 外法向
    2: np.array([-1, 0, 0]),   # 右面: 车头指向 Tag 的 -X
    3: np.array([0, 0, -1]),   # 后面: 车头 = 反外法向
}


def get_heading(pose):
    """从 Tag 位姿提取车头朝向（世界坐标系 XY 平面投影，单位向量）。"""
    h_local = _HEADING_IN_TAG_FRAME.get(pose["tag_id"], np.array([0, 0, 1]))
    h_world = pose["rotation"] @ h_local
    # 投影到水平面，归一化
    h_2d = h_world[:2]
    norm = np.linalg.norm(h_2d)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return h_2d / norm


# ======================================================================
# 目标定位（小车上的 Tag 即目标位置，无需偏移）
# ======================================================================

# 目标 Tag 集合
TARGET_TAG_IDS = {0, 1, 2, 3}


def tag_to_target_position(tag_id: int, tag_position: np.ndarray):
    """Tag 的世界位置直接作为目标位置。"""
    return tag_position.copy()


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
