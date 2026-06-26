#!/usr/bin/env python3
"""
Pi 端持续追踪服务 — 保持相机打开，循环检测
=============================================
启动后持续抓图→检测→输出 JSON，通过 TCP socket 发送结果。
PC 端只需连接一次，之后每次读一行 JSON 即可。

用法: python3 pi_tracker_server.py
"""

import cv2, time, json, sys, socket, threading
import numpy as np
from picamera2 import Picamera2
from pupil_apriltags import Detector

# 内参外参
CAMERAS = {
    "picam_1": {
        "K": np.array([[1050.3349,0,648.7089],[0,1048.6376,555.0087],[0,0,1]], dtype=np.float64),
        "dist": np.array([0.132095,-0.532177,0.011064,-0.003189,0.498587], dtype=np.float64),
        "type": "picamera",
    },
    "usb_cam_1": {
        "K": np.array([[1610.2608,0,962.8233],[0,1599.8428,804.8184],[0,0,1]], dtype=np.float64),
        "dist": np.array([0.150416,-0.251154,0.002832,0.000118,0.133763]),
        "type": "usb",
    },
}

import yaml
with open("extrinsics.yaml","r") as f: ext = yaml.safe_load(f)
for key in CAMERAS:
    CAMERAS[key]["R"] = np.array(ext[key]["R"])
    CAMERAS[key]["t"] = np.array(ext[key]["t"]).reshape(3,1)

TAG_SIZE = 0.135; TARGET_IDS = {0,1,2,3}
detector = Detector(families="tag36h11", quad_decimate=1.0)
clahe = cv2.createCLAHE(2.0, (8,8))

def solve_pose(d, K, dist, R_w2c, t_w2c):
    half = TAG_SIZE/2.0
    obj = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
    ok, rv, tv = cv2.solvePnP(obj, d.corners, K, dist)
    if not ok: return None
    Rt,_=cv2.Rodrigues(rv); tt=tv.reshape(3,1)
    Rc=R_w2c.T; tc=-Rc@t_w2c; tw=(Rc@tt+tc).flatten()
    P=R_w2c@tw.reshape(3,1)+t_w2c
    gsd=np.linalg.norm(P)/((K[0,0]+K[1,1])/2)*1000
    proj,_=cv2.projectPoints(obj.reshape(-1,1,3),rv,tv,K,dist)
    e=np.mean([np.linalg.norm(proj[i]-d.corners[i]) for i in range(4)])
    return {"tag_id":int(d.tag_id),"position":tw.tolist(),"gsd":round(float(gsd),2),
            "reproj_error":round(float(e),2),"decision_margin":round(float(d.decision_margin),1)}

def detect_from_frame(frame, K, dist, R, t, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    dets = detector.detect(gray)
    results = []
    t_cap = time.time()
    for d in dets:
        if d.tag_id in TARGET_IDS:
            pose = solve_pose(d, K, dist, R, t)
            if pose:
                pose["source"] = name; pose["t_capture"] = round(t_cap,3)
                results.append(pose)
    return results

# ===== 初始化相机（保持打开）=====
print("初始化相机...", file=sys.stderr)

# 启动前清理残留
import subprocess, os
subprocess.run(["pkill","-9","-f","picamera2"], capture_output=True)
subprocess.run(["pkill","-9","-f","libcamera"], capture_output=True)
time.sleep(2)

print("初始化相机...", file=sys.stderr)
picam = Picamera2(0)
picam.configure(picam.create_video_configuration(main={"size":(2028,1520),"format":"RGB888"}, buffer_count=1))
picam.start(); time.sleep(0.3)

# USB1
cap_usb = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap_usb.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
time.sleep(0.3)

# ===== TCP Server =====
PORT = 9999
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"服务已启动，端口 {PORT}", file=sys.stderr)

def handle_client(conn):
    print("客户端已连接", file=sys.stderr)
    while True:
        try:
            data = conn.recv(1024)  # 等待客户端发 tick
            if not data: break
        except: break

        t0 = time.time()
        results = []
        # 并行抓图
        arr = picam.capture_array()
        ret, usb_frame = cap_usb.read()
        # 检测
        cfg_p = CAMERAS["picam_1"]
        results.extend(detect_from_frame(arr, cfg_p["K"], cfg_p["dist"], cfg_p["R"], cfg_p["t"], "PiCam"))
        if ret:
            cfg_u = CAMERAS["usb_cam_1"]
            results.extend(detect_from_frame(usb_frame, cfg_u["K"], cfg_u["dist"], cfg_u["R"], cfg_u["t"], "USB1"))
        elapsed = (time.time()-t0)*1000
        msg = json.dumps({"results": results, "elapsed_ms": round(elapsed,1)}) + "\n"
        try: conn.sendall(msg.encode())
        except: break
    conn.close()

while True:
    conn, addr = server.accept()
    handle_client(conn)
