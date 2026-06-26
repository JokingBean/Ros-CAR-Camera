#!/usr/bin/env python3
"""
Pi 端追踪脚本 — 抓图 + 检测 + 输出 JSON 结果
=============================================
运行在树莓派上，被 PC 通过 SSH 调用。
捕获 PiCamera + USB1，检测小车 Tag，计算位姿，打印 JSON。"""

import cv2, time, json, sys, numpy as np
from picamera2 import Picamera2
from pupil_apriltags import Detector

# 内参外参（从 config.yaml / extrinsics.yaml 硬编码以加速启动）
CAMERAS = {
    "picam_1": {
        "K": np.array([[1064.8132,0,656.2857],[0,1056.9046,526.8922],[0,0,1]], dtype=np.float64),
        "dist": np.array([0.070544,-0.031997,-0.000403,0.000610,-0.052153], dtype=np.float64),
        "type": "picamera",
    },
    "usb_cam_1": {
        "K": np.array([[1610.2608,0,962.8233],[0,1599.8428,804.8184],[0,0,1]], dtype=np.float64),
        "dist": np.array([0.150416,-0.251154,0.002832,0.000118,0.133763], dtype=np.float64),
        "type": "usb",
    },
}

with open("extrinsics.yaml","r") as f:
    import yaml
    ext = yaml.safe_load(f)
for key in CAMERAS:
    CAMERAS[key]["R"] = np.array(ext[key]["R"])
    CAMERAS[key]["t"] = np.array(ext[key]["t"]).reshape(3,1)

TAG_SIZE = 0.135
TARGET_IDS = {0,1,2,3}
detector = Detector(families="tag36h11", quad_decimate=1.0)
clahe = cv2.createCLAHE(2.0, (8,8))

def solve_pose(detection, K, dist, R_w2c, t_w2c):
    """PnP 求解 Tag 世界位姿。"""
    half = TAG_SIZE / 2.0
    obj_pts = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(obj_pts, detection.corners, K, dist)
    if not success: return None
    R_t2c,_ = cv2.Rodrigues(rvec); t_t2c = tvec.reshape(3,1)
    R_c2w = R_w2c.T; t_c2w = -R_c2w @ t_w2c
    t_world = (R_c2w @ t_t2c + t_c2w).flatten()
    # GSD
    P = R_w2c @ t_world.reshape(3,1) + t_w2c
    dist_m = np.linalg.norm(P)
    focal = (K[0,0] + K[1,1]) / 2.0
    gsd = dist_m / focal * 1000.0
    # 重投影误差
    proj,_ = cv2.projectPoints(obj_pts.reshape(-1,1,3), rvec, tvec, K, dist)
    reproj = np.mean([np.linalg.norm(proj[i]-detection.corners[i]) for i in range(4)])
    return {"tag_id": int(detection.tag_id), "position": t_world.tolist(), "gsd": round(float(gsd),2), "reproj_error": round(float(reproj),2), "decision_margin": round(float(detection.decision_margin),1)}

def capture_and_detect(name, cfg):
    """抓一帧 + 检测小车 Tag。"""
    # 抓图
    if cfg["type"] == "picamera":
        picam = Picamera2(0)
        picam.configure(picam.create_video_configuration(main={"size":(1332,990),"format":"RGB888"}, buffer_count=2))
        picam.start(); time.sleep(0.3)
        frame = picam.capture_array()
        picam.close()
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        time.sleep(0.3)
        for _ in range(5): cap.read()
        ret, frame = cap.read(); cap.release()
        if not ret: return []

    # 检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    dets = detector.detect(gray)
    results = []
    for d in dets:
        if d.tag_id in TARGET_IDS:
            pose = solve_pose(d, cfg["K"], cfg["dist"], cfg["R"], cfg["t"])
            if pose:
                pose["source"] = name
                results.append(pose)
    return results

# 主流程
all_results = []
for name, cfg in CAMERAS.items():
    try:
        results = capture_and_detect(name, cfg)
        all_results.extend(results)
        print(f"[{name}] {len(results)} tags", file=sys.stderr)
    except Exception as e:
        print(f"[{name}] ERROR: {e}", file=sys.stderr)

print(json.dumps(all_results))
