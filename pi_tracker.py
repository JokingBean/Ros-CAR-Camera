#!/usr/bin/env python3
"""
Pi 端追踪服务 — 三相机 Tag 检测 + TCP 推送
============================================
在树莓派上运行，打开 3 台 USB 相机，实时检测立方体 AprilTag，
融合定位后通过 TCP Socket 发送 JSON 结果到 PC。

用法:
  python3 pi_tracker.py --pc-ip 192.168.1.100 --port 9527
"""

import cv2
import numpy as np
import yaml
import socket
import json
import time
import sys
import os
import argparse
import subprocess
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pupil_apriltags import Detector

# ==============================================================
# 配置
# ==============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.134         # 立方体 Tag 边长 (m)
TAG_HEIGHT = 0.212       # Tag 中心离地高度 (m)，立方体放在车上
CUBE_HALF = 0.125        # 立方体半边长 (m) = 25cm/2，Tag 居中贴在各面
# Tag ID → 在立方体面上的方向（立方体局部坐标系，+Y=车头方向）
TAG_FACE = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
GRID_STEP = 0.5          # 网格吸附步长 (m)
X_MIN, X_MAX = 0.0, 5.0
Y_MIN, Y_MAX = 0.0, 5.0
RESOLUTION = (2560, 1440)  # 全分辨率（ROI 裁剪保证速度）
FPS_TARGET = 20

# --------------------------------------------------------------
def load_configs():
    """加载所有配置文件。"""
    with open(os.path.join(ROOT, "cfg", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(ROOT, "cfg", "extrinsics.yaml"), "r") as f:
        ext = yaml.safe_load(f)
    with open(os.path.join(ROOT, "cfg", "floor_tags.yaml"), "r") as f:
        ft = yaml.safe_load(f)

    cam_params = {}
    scale_x = RESOLUTION[0] / 2560.0
    scale_y = RESOLUTION[1] / 1440.0

    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        K = np.array([[cm["fx"] * scale_x, 0, cm["cx"] * scale_x],
                      [0, cm["fy"] * scale_y, cm["cy"] * scale_y],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        if name in ext:
            R = np.array(ext[name]["R"], dtype=np.float64)
            t = np.array(ext[name]["t"], dtype=np.float64).reshape(3, 1)
        else:
            R = np.eye(3, dtype=np.float64)
            t = np.zeros((3, 1), dtype=np.float64)
        cam_params[name] = {
            "K": K, "dist": dist, "R": R, "t": t,
            "device": c["device"],
            "resolution": c.get("resolution", [2560, 1440]),
        }
    return cam_params


def _v4l2_preset(device_idx):
    """Linux V4L2 硬件参数预置：用 v4l2-ctl 直接写硬件寄存器。
    OpenCV 的 CAP_PROP_AUTO_EXPOSURE/AUTO_WB 在 V4L2 上映射不正确。
    """
    dev = f"/dev/video{device_idx}"
    if not os.path.exists(dev):
        return
    try:
        subprocess.run(
            f"v4l2-ctl -d {dev} --set-ctrl="
            f"auto_exposure=3,"
            f"white_balance_automatic=1,"
            f"brightness=28,"
            f"contrast=55,"
            f"sharpness=55,"
            f"gain=80",
            shell=True, capture_output=True, timeout=5)
    except Exception:
        pass


def open_cameras(cam_params):
    """打开所有相机，并为每台相机预创建 Detector 和 CLACHE。"""
    caps = {}
    detectors = {}
    clahes = {}
    for name, p in cam_params.items():
        device = p["device"]
        # Pi 上 OpenCV 不支持 /dev/videoX 路径，提取数字索引
        if isinstance(device, str) and "video" in device:
            idx = int(device.split("video")[-1])
        elif isinstance(device, str) and device.isdigit():
            idx = int(device)
        else:
            idx = device
        # 先用 v4l2-ctl 预置 V4L2 硬件参数（OpenCV 映射不正确）
        _v4l2_preset(idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  [{name}] 无法打开 {device} (idx={idx})")
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # OpenCV 打开相机后可能重置参数，再设一次 v4l2-ctl
        _v4l2_preset(idx)
        # 丢弃缓冲帧
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret and frame.mean() > 5:
            caps[name] = cap
            detectors[name] = Detector(families="tag36h11", quad_decimate=1.0)
            clahes[name] = cv2.createCLAHE(2.5, (8, 8))
            print(f"  [{name}] {device} -> {frame.shape[1]}x{frame.shape[0]} mean={frame.mean():.0f}")
        else:
            print(f"  [{name}] {device} 画面异常, 跳过")
            cap.release()
    return caps, detectors, clahes


def _process_detections(dets, K, dist, R, t):
    """将检测结果转为世界坐标，含质量过滤（面积/倾斜角/重投影误差）。"""
    results = []
    half = TAG_SIZE / 2.0
    obj_pts = np.array([
        [-half, -half, 0], [half, -half, 0],
        [half, half, 0], [-half, half, 0]
    ], dtype=np.float64)

    for d in dets:
        if d.tag_id not in TARGET_IDS:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_pts, d.corners, K, dist)
        if not ok:
            continue
        # solvePnP 对平面 Tag 有 180° 歧义性，用 Tag 到相机的方向判断是否翻转
        R_tag2cam, _ = cv2.Rodrigues(rvec)
        # Tag Z 轴在相机空间指向 Tag 面的朝向，应指向相机原点 (0,0,0)
        # 相机原点到 Tag 的方向 = -tvec，与 Z 轴点积 > 0 表示指向相机
        if np.dot(R_tag2cam[:, 2], -tvec.flatten()) < 0:  # Z 轴背离相机 → 翻转
            R_tag2cam[:, 2] = -R_tag2cam[:, 2]
            R_tag2cam[:, 0] = -R_tag2cam[:, 0]  # 保持右手系
            rvec, _ = cv2.Rodrigues(R_tag2cam)
        t_tag2cam = tvec.reshape(3, 1)
        R_c2w = R.T
        t_c2w = -R_c2w @ t
        tw = (R_c2w @ t_tag2cam + t_c2w).flatten()

        # 基于几何的车头方向：Tag 到相机的方向 + Tag 在立方体上的局部偏移
        cam_pos = t_c2w.flatten()  # 相机世界坐标
        dx = cam_pos[0] - tw[0]
        dy = cam_pos[1] - tw[1]
        geom_yaw = float(np.arctan2(dy, dx))  # Tag→相机的方向
        # TAG_FACE 定义 Tag 在立方体上的法线方向（+Y=车头）
        TAG_FACE_YAW = {0: -np.pi/2, 1: np.pi, 2: np.pi/2, 3: 0.0}  # 面法线→车头
        car_yaw_geom = geom_yaw + TAG_FACE_YAW.get(d.tag_id, 0.0)

        # 航向角：Tag 面法向量在 XY 平面的投影方向
        R_tag2w = R_c2w @ R_tag2cam
        nx, ny = R_tag2w[0, 2], R_tag2w[1, 2]  # Tag Z轴在世界的投影
        yaw = float(np.arctan2(ny, nx))

        P_cam = R @ tw.reshape(3, 1) + t
        dist_cam = np.linalg.norm(P_cam)
        focal = (K[0, 0] + K[1, 1]) / 2.0
        gsd = dist_cam / focal * 1000.0
        results.append({
            "tag_id": d.tag_id,
            "center_xy": [float(tw[0]), float(tw[1])],
            "cube_xy": [float(tw[0]), float(tw[1])],
            "tag_3d": [float(tw[0]), float(tw[1]), float(tw[2])],
            "gsd": round(float(gsd), 2),
            "margin": float(d.decision_margin),
            "yaw": round(yaw, 4),
            "car_yaw_geom": round(car_yaw_geom, 4),
        })
    return results


# 共享 ROI 状态：基于世界坐标的跨相机 ROI
# {"world_xy": (x, y), "yaw": 0.0, "miss": 0, "rois": {...}}
_shared_roi = {}

ROI_BASE = 300       # 初始 ROI 边长 (像素)
ROI_EXPAND = 200     # 每次找不到扩大多少
ROI_MAX_MISS = 4     # 连续 miss 几次后回退全图
PREDICT_STEP = 0.10  # 丢失时沿航向每次预测前进距离 (m)


def _world_to_image(x, y, z, K, R, t):
    """世界坐标 → 图像坐标。返回 (u, v) 或 None。"""
    P_w = np.array([[x], [y], [z]], dtype=np.float64)
    P_c = R @ P_w + t
    if P_c[2, 0] <= 0:
        return None
    uv = K @ P_c
    return (int(uv[0, 0] / uv[2, 0]), int(uv[1, 0] / uv[2, 0]))


def _calc_rois(world_xy, yaw, cam_params):
    """根据世界坐标+航向计算所有相机的 ROI（优先覆盖车头和车尾）。"""
    x, y = world_xy
    # 车前位置（Tag 3 面，车头方向）和车尾位置（Tag 1 面）
    fx = x + CUBE_HALF * math.cos(yaw)
    fy = y + CUBE_HALF * math.sin(yaw)
    bx = x - CUBE_HALF * math.cos(yaw)
    by = y - CUBE_HALF * math.sin(yaw)
    rois = {}
    for name, p in cam_params.items():
        uv_f = _world_to_image(fx, fy, TAG_HEIGHT, p["K"], p["R"], p["t"])
        uv_b = _world_to_image(bx, by, TAG_HEIGHT, p["K"], p["R"], p["t"])
        res = p["resolution"]
        # 合并前后 ROI，取最小外接矩形
        pts = []
        for uv in (uv_f, uv_b):
            if uv and 0 <= uv[0] < res[0] and 0 <= uv[1] < res[1]:
                pts.append(uv)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        half = ROI_BASE // 2
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(res[0], cx + half), min(res[1], cy + half)
        rois[name] = (x0, y0, x1 - x0, y1 - y0)
    return rois


def update_shared_roi(world_xy, yaw, cam_params):
    """定位成功：记录位置+航向，刷新所有相机 ROI。"""
    global _shared_roi
    _shared_roi["world_xy"] = world_xy
    _shared_roi["yaw"] = yaw
    _shared_roi["miss"] = 0
    _shared_roi["rois"] = _calc_rois(world_xy, yaw, cam_params)


def on_roi_miss(cam_params):
    """丢失 Tag：沿车头方向预测位置，重新算 ROI。连续 miss 太多则回退全图。"""
    global _shared_roi
    miss = _shared_roi.get("miss", 0) + 1
    _shared_roi["miss"] = miss

    if miss <= ROI_MAX_MISS:
        yaw = _shared_roi.get("yaw", 0.0)
        old_xy = _shared_roi.get("world_xy", (0, 0))
        # 沿车头方向前进
        step = PREDICT_STEP * miss
        new_x = old_xy[0] + step * np.cos(yaw)
        new_y = old_xy[1] + step * np.sin(yaw)
        _shared_roi["rois"] = _calc_rois((new_x, new_y), cam_params)
    else:
        # 回退全图
        _shared_roi["rois"] = {}


def _wb_correct(img):
    """灰度世界白平衡：按 BGR 均值缩放各通道，消除色偏。"""
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    avg = (mb + mg + mr) / 3.0
    if avg < 1:
        return img
    scale_b = avg / mb if mb > 0 else 1.0
    scale_g = avg / mg if mg > 0 else 1.0
    scale_r = avg / mr if mr > 0 else 1.0
    # 限制缩放范围，避免极端偏色
    scale_b = np.clip(scale_b, 0.7, 1.3)
    scale_g = np.clip(scale_g, 0.7, 1.3)
    scale_r = np.clip(scale_r, 0.7, 1.3)
    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def detect_cube_tags_roi(img, K, dist, R, t, detector, clahe, name):
    """跨相机 ROI 检测：根据相机能否看到当前位置决定搜索策略。"""
    global _shared_roi
    h, w = img.shape[:2]

    # ---- 首次定位 / 完全丢失 → 全图搜索 ----
    if "world_xy" not in _shared_roi:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_s = clahe.apply(gray)
        dets = detector.detect(gray_s)
        found = [d for d in dets if d.tag_id in TARGET_IDS]
        dets_out = _process_detections(found, K, dist, R, t)
        return dets_out

    # ---- 已有定位结果 ----
    rois = _shared_roi.get("rois", {})

    # 该相机看不到当前位置 → 直接跳过（不做全图搜索拖慢帧率）
    if name not in rois:
        return []

    # ---- ROI 搜索 ----
    x, y, rw, rh = rois[name]
    x = max(0, x); y = max(0, y)
    rw = min(w - x, rw); rh = min(h - y, rh)
    if rw <= 20 or rh <= 20:
        return []
    crop = img[y:y + rh, x:x + rw]
    if crop.size == 0:
        return []
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_gray = clahe.apply(crop_gray)
    dets = detector.detect(crop_gray)
    for d in dets:
        d.corners[:, 0] += x
        d.corners[:, 1] += y
        d.center = (d.center[0] + x, d.center[1] + y)
    found = [d for d in dets if d.tag_id in TARGET_IDS]
    if found:
        dets_out = _process_detections(found, K, dist, R, t)
        return dets_out
    return []


def grid_snap(x, y, step=GRID_STEP):
    """吸附到最近网格点。"""
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))


# ==============================================================
# 多相机融合
# ==============================================================

def fuse_positions(all_results):
    """所有 Tag 按 margin 权重直接加权平均，不分组、不取中位数。"""
    if not all_results:
        return None

    total_tags = len(all_results)

    # 每个 tag 的 margin 作为权重（置信度越高权重越大）
    pts = np.array([r.get("cube_xy", r["center_xy"]) for r in all_results])
    margins = np.array([max(r.get("margin", 0), 0.1) for r in all_results], dtype=float)
    weights = margins / margins.sum()

    fused_xy = np.average(pts, axis=0, weights=weights).tolist()

    return {
        "fused_xy": fused_xy,
        "n_obs": total_tags,
        "n_cam": len(set(r["camera"] for r in all_results)),
    }


# ==============================================================
# TCP 发送
# ==============================================================

def connect_to_pc(host, port, retry_interval=3.0):
    """连接到 PC 的 TCP 服务器，持续重试直到成功。"""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[TCP] 已连接到 PC {host}:{port}")
            return sock
        except (ConnectionRefusedError, OSError, socket.timeout) as e:
            print(f"[TCP] 连接失败 ({e})，{retry_interval}s 后重试...")
            time.sleep(retry_interval)


def send_result(sock, data):
    """发送一行 JSON 到 PC。"""
    try:
        line = json.dumps(data, ensure_ascii=False) + "\n"
        sock.sendall(line.encode("utf-8"))
        return True
    except (BrokenPipeError, OSError) as e:
        print(f"[TCP] 发送失败: {e}")
        return False


# ==============================================================
# 主循环
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Pi 端追踪服务")
    parser.add_argument("--pc-ip", required=True, help="PC 的 IP 地址")
    parser.add_argument("--port", type=int, default=9527, help="TCP 端口 (默认 9527)")
    parser.add_argument("--config-dir", default=None, help="配置文件目录 (默认项目根/cfg)")
    args = parser.parse_args()

    # 可切换配置目录
    if args.config_dir:
        global ROOT
        ROOT = args.config_dir

    print("=" * 50)
    print("  Pi 追踪服务 — 三相机 Tag 检测")
    print(f"  目标 PC: {args.pc_ip}:{args.port}")
    print("=" * 50)

    # --- 加载配置 ---
    print("\n[1/3] 加载配置...")
    cam_params = load_configs()
    print(f"  已加载 {len(cam_params)} 台相机参数")

    # --- 打开相机 ---
    print("\n[2/3] 打开相机...")
    caps, detectors, clahes = open_cameras(cam_params)
    if not caps:
        print("错误: 没有可用的相机")
        sys.exit(1)
    print(f"  成功打开 {len(caps)} 台相机")

    # --- 连接 PC ---
    print(f"\n[3/3] 连接 PC {args.pc_ip}:{args.port}...")
    sock = connect_to_pc(args.pc_ip, args.port)

    # --- 主循环 ---
    print("\n开始追踪 (Ctrl+C 停止)...\n")
    fps_hist = deque(maxlen=30)
    frame_count = 0
    smooth_yaw = None  # 平滑后的航向

    try:
        # 使用线程池：抓帧和检测都并行
        with ThreadPoolExecutor(max_workers=3) as executor:
            while True:
                t_loop = time.time()

                # 1. 并行抓取所有相机帧
                capture_futures = {
                    executor.submit(lambda c=c: c.read()): name
                    for name, c in caps.items()
                }
                frames = {}
                for fut in as_completed(capture_futures):
                    name = capture_futures[fut]
                    ret, frame = fut.result()
                    if ret:
                        frames[name] = frame

                # 2. 并行检测 Tag
                all_results = []
                if frames:
                    detect_futures = {}
                    for name, frame in frames.items():
                        p = cam_params[name]
                        fut = executor.submit(
                            detect_cube_tags_roi,
                            frame, p["K"], p["dist"], p["R"], p["t"],
                            detectors[name], clahes[name], name)
                        detect_futures[fut] = name

                    for fut in as_completed(detect_futures):
                        name = detect_futures[fut]
                        try:
                            tags = fut.result()
                        except Exception as e:
                            print(f"  [{name}] 检测异常: {e}")
                            continue
                        for t in tags:
                            t["camera"] = name
                        all_results.extend(tags)

                # 3. 融合定位
                if all_results:
                    fused = fuse_positions(all_results)
                    fx, fy = fused["fused_xy"]
                    n_cam = fused["n_cam"]

                    # 航向：基于 Tag 到相机的几何方向 + Tag 在立方体上的局部偏移
                    car_yaws = []
                    tag_yaws_str = []
                    TAG_FACE_YAW = {0: -np.pi/2, 1: np.pi, 2: np.pi/2, 3: 0.0}
                    for r in all_results:
                        tid = r.get("tag_id")
                        raw_yaw = r.get("yaw", 0.0)
                        car_yaw = raw_yaw + TAG_FACE_YAW.get(tid, 0.0)
                        car_yaws.append(car_yaw)
                        tag_yaws_str.append(f"T{tid}({raw_yaw*180/np.pi:.0f}°+{TAG_FACE_YAW[tid]*180/np.pi:.0f}°)")
                    if car_yaws:
                        # 归一化到 [-π, π] 后取初版中位数
                        car_yaws_norm = [math.atan2(math.sin(v), math.cos(v)) for v in car_yaws]
                        median_cand = float(np.median(car_yaws_norm))
                        # 剔除偏离中位数 > 45° 的异常值，重算
                        filtered = [v for v in car_yaws_norm
                                    if abs(math.atan2(math.sin(v - median_cand),
                                                      math.cos(v - median_cand))) < np.pi / 4]
                        if filtered:
                            fused_yaw = float(np.median(filtered))
                        else:
                            fused_yaw = median_cand
                    else:
                        fused_yaw = 0.0

                    smooth_yaw = fused_yaw

                    # 直接使用融合值，不做时序平滑/死区锁定
                    sx, sy = fx, fy
                    smooth = np.array([sx, sy])

                    # 更新 ROI（基于最新融合值+航向，优先车头车尾）
                    update_shared_roi((sx, sy), smooth_yaw if smooth_yaw is not None else 0.0, cam_params)

                    # 网格吸附 + 误差
                    gx, gy = grid_snap(smooth[0], smooth[1])
                    err = np.linalg.norm([smooth[0] - gx, smooth[1] - gy]) * 100

                    # FPS
                    elapsed = (time.time() - t_loop) * 1000
                    if elapsed > 0:
                        fps_hist.append(1000 / elapsed)
                    fps = np.mean(fps_hist) if fps_hist else 0

                    # 构建发送数据
                    data = {
                        "t": int(time.time() * 1000),
                        "x": round(float(smooth[0]), 3),
                        "y": round(float(smooth[1]), 3),
                        "grid_x": float(gx),
                        "grid_y": float(gy),
                        "err_cm": round(float(err), 1),
                        "fps": round(float(fps), 1),
                        "yaw": round(float(smooth_yaw), 4),
                        "n_cams": len(frames),
                        "n_obs": fused["n_obs"],
                        "raw_yaws": tag_yaws_str,
                        "raw": all_results,
                    }

                    # 发送
                    if not send_result(sock, data):
                        print("[TCP] 连接断开，重连中...")
                        sock.close()
                        sock = connect_to_pc(args.pc_ip, args.port)

                    # 本地打印（每帧输出，含原始融合值和各相机位置）
                    frame_count += 1
                    if frame_count % 5 == 0:
                        tags_str = " ".join(f"{r['camera']}T{r['tag_id']}" for r in all_results)
                        # 各相机位置汇总
                        cam_positions = {}
                        for r in all_results:
                            cam = r["camera"]
                            xy = r["center_xy"]
                            tid = r["tag_id"]
                            if cam not in cam_positions:
                                cam_positions[cam] = []
                            cam_positions[cam].append(f"T{tid}({xy[0]:.3f},{xy[1]:.3f})")
                        cam_str = "  ".join(f"{c}:{','.join(v)}" for c, v in cam_positions.items())
                        print(f"\r  XY=({smooth[0]:.3f},{smooth[1]:.3f}) "
                              f"RAW=({fx:.3f},{fy:.3f}) "
                              f"yaw={fused_yaw*180/np.pi:.1f}° "
                              f"[{cam_str}]  "
                              f"yaw_dbg=[{','.join(tag_yaws_str)}]  "
                              f"FPS={fps:.1f}  ",
                              end="", flush=True)
                else:
                    # 无检测，沿车头方向预测位置
                    on_roi_miss(cam_params)
                    elapsed = (time.time() - t_loop) * 1000
                    if elapsed > 0:
                        fps_hist.append(1000 / elapsed)
                    fps = np.mean(fps_hist) if fps_hist else 0

                    data = {
                        "t": int(time.time() * 1000),
                        "x": -99.0, "y": -99.0, "z": -99.0,
                        "grid_x": -99.0, "grid_y": -99.0, "err_cm": -99.0,
                        "fps": round(float(fps), 1),
                        "n_cams": len(frames),
                        "n_obs": 0,
                    }
                    if not send_result(sock, data):
                        sock.close()
                        sock = connect_to_pc(args.pc_ip, args.port)

                    frame_count += 1
                    if frame_count % 5 == 0:
                        print(f"\r  等待立方体... FPS={fps:.1f} [无检测]  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n停止追踪")
    finally:
        sock.close()
        for cap in caps.values():
            cap.release()
        print("已释放所有资源")


if __name__ == "__main__":
    main()
