"""
立方体 Tag 定位 — solvePnP + homography
=======================================
从相机图像中检测立方体 Tag (ID 0-3)，通过 solvePnP 计算 3D 世界坐标。
"""

import cv2
import numpy as np
from pupil_apriltags import Detector

TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.135
GRID_STEP = 0.5
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = 0.0, 5.0


def detect_cube_extrinsics(img, K, dist, R, t):
    """外参 solvePnP 定位立方体 Tag。

    参数:
        img — BGR 图像
        K, dist, R, t — 相机内参和外参

    返回:
        [{tag_id, tag_3d, center_xy, gsd, diag_px, margin}, ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 0.5 if max(img.shape) > 2000 else 1.0
    gray_s = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
    gray_s = cv2.createCLAHE(2.0, (8, 8)).apply(gray_s)

    detector = Detector(families="tag36h11", quad_decimate=1.0)
    dets = detector.detect(gray_s)
    if scale != 1.0:
        for d in dets:
            d.corners /= scale
            d.center = (d.center[0] / scale, d.center[1] / scale)

    results = []
    half = TAG_SIZE / 2.0
    obj_pts = np.array([[-half, -half, 0], [half, -half, 0],
                         [half, half, 0], [-half, half, 0]], dtype=np.float64)

    for d in dets:
        if d.tag_id not in TARGET_IDS:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_pts, d.corners, K, dist)
        if not ok:
            continue

        R_tag2cam, _ = cv2.Rodrigues(rvec)
        t_tag2cam = tvec.reshape(3, 1)
        R_c2w = R.T
        t_c2w = -R_c2w @ t
        tw = (R_c2w @ t_tag2cam + t_c2w).flatten()

        P = R @ tw.reshape(3, 1) + t
        gsd = np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000

        results.append({
            "tag_id": d.tag_id,
            "tag_3d": [float(tw[0]), float(tw[1]), float(tw[2])],
            "center_xy": [float(tw[0]), float(tw[1])],
            "gsd": round(float(gsd), 2),
            "diag_px": float(np.linalg.norm(d.corners[0] - d.corners[2])),
            "margin": float(d.decision_margin),
        })
    return results


def grid_snap(x, y, step=GRID_STEP):
    """吸附到最近网格点。"""
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))
