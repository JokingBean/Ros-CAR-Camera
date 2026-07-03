#!/usr/bin/env python3
"""
低延时实时追踪 — PC 端
======================
Pi 端: SSH 调用 pi_tracker.py → JSON 结果（无图片传输）
本机: usb3 直接抓图检测
融合: GSD 加权 → 打印 + 可选 BEV 叠加
"""

import cv2, time, json, sys, os, subprocess
import numpy as np
from pathlib import Path
from pupil_apriltags import Detector

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"

# usb3 内参外参（本机 PC）
K2 = np.array([[1997.5587,0,1203.9179],[0,2004.3731,784.2230],[0,0,1]], dtype=np.float64)
D2 = np.array([0.08367,-0.15649,0.00321,-0.00835,0.11271], dtype=np.float64)
import yaml
with open("extrinsics.yaml","r") as f: ext = yaml.safe_load(f)
R2 = np.array(ext["usb3"]["R"])
t2 = np.array(ext["usb3"]["t"]).reshape(3,1)

TAG_SIZE = 0.135
TARGET_IDS = {0,1,2,3}
detector = Detector(families="tag36h11", quad_decimate=1.0)
clahe = cv2.createCLAHE(2.0, (8,8))

# 朝向
_HEADING_IN_TAG = {0:np.array([1,0,0]),1:np.array([0,0,1]),2:np.array([-1,0,0]),3:np.array([0,0,-1])}

def capture_usb2():
    for idx in [1]:  # skip idx=0
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(0.2)
        for _ in range(5): cap.read()
        ret, frame = cap.read(); cap.release()
        if ret and frame.mean() > 10: return frame
    return None

def solve_pose(d, K, dist, R_w2c, t_w2c):
    half = TAG_SIZE/2.0
    obj = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
    ok, rv, tv = cv2.solvePnP(obj, d.corners, K, dist)
    if not ok: return None
    Rt,_=cv2.Rodrigues(rv); tt=tv.reshape(3,1)
    Rc=R_w2c.T; tc=-Rc@t_w2c
    tw=(Rc@tt+tc).flatten()
    P=R_w2c@tw.reshape(3,1)+t_w2c
    gsd=np.linalg.norm(P)/((K[0,0]+K[1,1])/2)*1000
    proj,_=cv2.projectPoints(obj.reshape(-1,1,3),rv,tv,K,dist)
    e=np.mean([np.linalg.norm(proj[i]-d.corners[i]) for i in range(4)])
    h_local=_HEADING_IN_TAG.get(d.tag_id,np.array([0,0,1]))
    h_w=Rc@Rt@h_local; h_2d=h_w[:2]; h_2d/=np.linalg.norm(h_2d)
    return {"tag_id":int(d.tag_id),"position":tw,"gsd":round(float(gsd),2),"reproj_error":round(float(e),2),"heading":h_2d.tolist()}

def get_pi_results(ssh):
    """通过 SSH 执行 pi_tracker.py，返回结果列表。"""
    # 上传脚本（首次）
    if not hasattr(get_pi_results, "_uploaded"):
        sftp = ssh.open_sftp()
        with open("pi_tracker.py","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/pi_tracker.py")
        # 也上传 extrinsics.yaml
        with open("extrinsics.yaml","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/extrinsics.yaml")
        sftp.close()
        get_pi_results._uploaded = True

    stdin, stdout, stderr = ssh.exec_command(
        "cd /home/pi/UwbCamera && python3 pi_tracker.py 2>/dev/null", timeout=8)
    out = stdout.read().decode().strip()
    if out:
        try:
            return json.loads(out)
        except: pass
    return []

def fuse(results):
    """GSD 加权融合，返回 (position, heading, gsd)"""
    if not results: return None
    weights = np.array([1.0/max(r["gsd"],0.01) for r in results])
    weights /= weights.sum()
    pos = np.zeros(3)
    for w, r in zip(weights, results):
        pos += w * np.array(r["position"])
    headings = [np.array(r.get("heading",[0,1])) for r in results]
    hw = np.zeros(2)
    for w, h in zip(weights, headings):
        hw += w * h
    hw /= np.linalg.norm(hw)
    best = min(results, key=lambda r: r["gsd"])
    return {"position": pos, "heading": hw, "gsd": best["gsd"], "sources": list(set(r.get("source","?") for r in results))}

def main():
    print("低延时实时追踪 (Ctrl+C 停止)")
    print("连接树莓派...")

    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
    except Exception as e:
        print(f"树莓派连接失败: {e}")
        sys.exit(1)
    print("已连接\n")

    cycle = 0
    try:
        while True:
            t0 = time.time()
            cycle += 1

            # 并行：Pi 处理 + USB2 处理
            pi_results = get_pi_results(ssh)

            frame = capture_usb2()
            usb2_results = []
            if frame is not None:
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
                gray = clahe.apply(gray)
                for d in detector.detect(gray):
                    if d.tag_id in TARGET_IDS:
                        d.corners *= 2.0; d.center = (d.center[0]*2, d.center[1]*2)
                        p = solve_pose(d, K2, D2, R2, t2)
                        if p: p["source"]="USB2"; usb2_results.append(p)

            # 合并 + 融合
            all_r = pi_results + usb2_results
            fused = fuse(all_r)

            elapsed = (time.time() - t0) * 1000
            if fused:
                p = fused["position"]
                hdg = np.degrees(np.arctan2(fused["heading"][0], fused["heading"][1]))
                srcs = ",".join(fused["sources"])
                print(f"[{cycle:04d}] ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) hdg={hdg:.0f}deg gsd={fused['gsd']:.1f}mm [{srcs}] ({elapsed:.0f}ms)")
            else:
                print(f"[{cycle:04d}] 未检测到小车 ({elapsed:.0f}ms)")

    except KeyboardInterrupt:
        print("\n停止")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
