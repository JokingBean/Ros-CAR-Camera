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
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pupil_apriltags import Detector

# ==============================================================
# 配置
# ==============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.135         # 立方体 Tag 边长 (m)
TAG_HEIGHT = 0.25        # Tag 中心离地高度 (m)，立方体放在车上
GRID_STEP = 0.5          # 网格吸附步长 (m)
X_MIN, X_MAX = 0.0, 5.0
Y_MIN, Y_MAX = 0.0, 5.0
RESOLUTION = (1280, 720)   # 快采集
CALIB_RESOLUTION = (2560, 1440)  # 内参标定分辨率
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
    scale_x = RESOLUTION[0] / CALIB_RESOLUTION[0]
    scale_y = RESOLUTION[1] / CALIB_RESOLUTION[1]
    print(f"  内参缩放: {scale_x:.2f}x ({CALIB_RESOLUTION[0]}x{CALIB_RESOLUTION[1]} → {RESOLUTION[0]}x{RESOLUTION[1]})")

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
            "brightness": c.get("brightness"),
            "contrast": c.get("contrast"),
            "saturation": c.get("saturation"),
            "gain": c.get("gain"),
            "white_balance": c.get("white_balance"),
        }
    return cam_params


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
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  [{name}] 无法打开 {device} (idx={idx})")
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 从 config 读取相机参数
        bright = p.get("brightness", 28)
        contrast = p.get("contrast", 40)
        sat = p.get("saturation", 64)
        gain = p.get("gain", 16)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, sat)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        wb = p.get("white_balance", 5000)
        try:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        except Exception:
            pass
        time.sleep(0.3)
        # 丢弃缓冲帧
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret and frame.mean() > 5:
            caps[name] = cap
            detectors[name] = Detector(families="tag36h11", quad_decimate=1.0)
            clahes[name] = cv2.createCLAHE(2.0, (8, 8))
            print(f"  [{name}] {device} -> {frame.shape[1]}x{frame.shape[0]} mean={frame.mean():.0f}")
        else:
            print(f"  [{name}] {device} 画面异常, 跳过")
            cap.release()
    return caps, detectors, clahes


def _process_detections(dets, K, dist, R, t):
    """将检测结果转为世界坐标，包含航向角 yaw。"""
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
        R_tag2cam, _ = cv2.Rodrigues(rvec)
        t_tag2cam = tvec.reshape(3, 1)
        R_c2w = R.T
        t_c2w = -R_c2w @ t
        tw = (R_c2w @ t_tag2cam + t_c2w).flatten()

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
            "tag_3d": [float(tw[0]), float(tw[1]), float(tw[2])],
            "gsd": round(float(gsd), 2),
            "margin": float(d.decision_margin),
            "yaw": round(yaw, 4),
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


def _calc_rois(world_xy, cam_params):
    """根据世界坐标计算所有相机的 ROI（考虑 Tag 离地高度）。"""
    x, y = world_xy
    rois = {}
    for name, p in cam_params.items():
        uv = _world_to_image(x, y, TAG_HEIGHT, p["K"], p["R"], p["t"])
        res = p["resolution"]
        if uv and 0 <= uv[0] < res[0] and 0 <= uv[1] < res[1]:
            half = ROI_BASE // 2
            x0, y0 = max(0, uv[0] - half), max(0, uv[1] - half)
            x1, y1 = min(res[0], uv[0] + half), min(res[1], uv[1] + half)
            rois[name] = (x0, y0, x1 - x0, y1 - y0)
    return rois


def update_shared_roi(world_xy, yaw, cam_params):
    """定位成功：记录位置+航向，刷新所有相机 ROI。"""
    global _shared_roi
    _shared_roi["world_xy"] = world_xy
    _shared_roi["yaw"] = yaw
    _shared_roi["miss"] = 0
    _shared_roi["rois"] = _calc_rois(world_xy, cam_params)


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
    """跨相机 ROI 检测：有预测位置 → 搜 ROI。首次或 miss>MAX 回退全图。"""
    global _shared_roi
    h, w = img.shape[:2]

    # 简单白平衡矫正：灰度世界假设，消除黄偏
    img = _wb_correct(img)

    # ---- 尝试 ROI ----
    rois = _shared_roi.get("rois", {})
    miss = _shared_roi.get("miss", 0)
    if name in rois and miss <= ROI_MAX_MISS:
        x, y, rw, rh = rois[name]
        # 裁剪到图像边界内
        x = max(0, x); y = max(0, y)
        rw = min(w - x, rw); rh = min(h - y, rh)
        if rw > 20 and rh > 20:
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
                return _process_detections(found, K, dist, R, t)
            return []

    # ---- 全图检测 ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_s = clahe.apply(gray)
    dets = detector.detect(gray_s)

    found = [d for d in dets if d.tag_id in TARGET_IDS]
    return _process_detections(found, K, dist, R, t)


def grid_snap(x, y, step=GRID_STEP):
    """吸附到最近网格点。"""
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))


def fuse_positions(all_results):
    """多相机加权融合定位：GSD 倒数加权 + 中位数。"""
    if not all_results:
        return None

    good = [r for r in all_results if r.get("margin", 0) >= 20]
    if not good:
        good = all_results

    xys = np.array([r["center_xy"] for r in good])
    gsds = np.array([r.get("gsd", 1.0) for r in good])
    weights = 1.0 / np.maximum(gsds, 0.01)
    weights /= weights.sum()

    weighted_xy = np.average(xys, axis=0, weights=weights)
    median_xy = np.median(xys, axis=0)

    return {
        "fused_xy": [float(median_xy[0]), float(median_xy[1])],
        "weighted_xy": [float(weighted_xy[0]), float(weighted_xy[1])],
        "n_obs": len(good),
        "n_total": len(all_results),
    }


# ==============================================================
# EKF 状态估计器 — 替代滑动窗口平滑
# ==============================================================

class CarEKF:
    """4 状态 EKF：[x, y, yaw, v]，恒速运动模型，相机观测 [x, y, yaw]。
    
    原理：
      - 预测 (predict)：根据速度 v 和航向 yaw 推算下一帧位置
      - 更新 (update)：用相机 GSD 加权观测修正预测
      - 协方差 P 自动平衡预测和观测的可信度
    
    效果：轨迹平滑无跳变，噪声抑制远优于简单滑动平均。
    """
    def __init__(self):
        self.initialized = False
        self.x = np.zeros((4, 1), dtype=np.float64)   # [x, y, yaw, v]
        self.P = np.eye(4) * 0.5                       # 状态协方差
        self.Q = np.diag([0.01, 0.01, 0.02, 0.05])    # 过程噪声（运动不确定性）
        self.R = np.diag([0.02, 0.02, 0.05])           # 观测噪声（相机不确定性）
        self.last_t = None
        self._H = np.zeros((3, 4))                     # 观测矩阵: 只看 [x, y, yaw]
        self._H[0, 0] = 1; self._H[1, 1] = 1; self._H[2, 2] = 1
        self._I = np.eye(4)

    def predict(self, dt):
        """恒速运动模型预测：x += v*cos(yaw)*dt, y += v*sin(yaw)*dt。"""
        if dt <= 0 or not self.initialized:
            return
        v = self.x[3, 0]
        yaw = self.x[2, 0]
        self.x[0, 0] += v * np.cos(yaw) * dt
        self.x[1, 0] += v * np.sin(yaw) * dt
        # Jacobian F
        F = np.eye(4)
        F[0, 2] = -v * np.sin(yaw) * dt
        F[0, 3] = np.cos(yaw) * dt
        F[1, 2] = v * np.cos(yaw) * dt
        F[1, 3] = np.sin(yaw) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """EKF 更新，带航向异常检测：yaw 跳变 > 90° 时只更新位置，跳过 yaw。"""
        if len(z) < 3:
            return
        z = np.array(z[:3], dtype=np.float64).reshape(3, 1)

        # 航向异常检测：观测 yaw 与预测 yaw 差 > 90° → 只更新 XY
        yaw_diff = z[2, 0] - self.x[2, 0]
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        skip_yaw = abs(yaw_diff) > np.pi / 2

        if skip_yaw:
            # 只用 XY 更新，跳过 yaw
            H_xy = self._H[:2, :]  # 2x4
            z_xy = z[:2]
            y_xy = z_xy - H_xy @ self.x
            R_xy = self.R[:2, :2]
            S_xy = H_xy @ self.P @ H_xy.T + R_xy
            K_xy = self.P @ H_xy.T @ np.linalg.inv(S_xy)
            self.x = self.x + K_xy @ y_xy
            self.P = (self._I - K_xy @ H_xy) @ self.P
        else:
            # 正常更新 XY + yaw
            y = z - self._H @ self.x
            y[2, 0] = yaw_diff  # 用已归一化的差
            S = self._H @ self.P @ self._H.T + self.R
            K = self.P @ self._H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.x[2, 0] = np.arctan2(np.sin(self.x[2, 0]), np.cos(self.x[2, 0]))
            self.P = (self._I - K @ self._H) @ self.P

    def init_state(self, x, y, yaw, v=0.0):
        """首次观测初始化状态。"""
        self.x = np.array([[x], [y], [yaw], [v]], dtype=np.float64)
        self.P = np.eye(4) * 0.5
        self.initialized = True
        self.last_t = None

    def get_state(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])


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
    ekf = CarEKF()
    frame_count = 0

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

                    # 航向：优先 tag3(车头) → tag1+180°(车尾反向)
                    tag3 = [r for r in all_results if r.get("tag_id") == 3]
                    tag1 = [r for r in all_results if r.get("tag_id") == 1]
                    if tag3:
                        fused_yaw = tag3[0].get("yaw", 0.0)
                    elif tag1:
                        fused_yaw = tag1[0].get("yaw", 0.0) + np.pi
                    elif all_results:
                        fused_yaw = all_results[0].get("yaw", 0.0)
                    else:
                        fused_yaw = 0.0
                    fused_yaw = float(np.arctan2(np.sin(fused_yaw), np.cos(fused_yaw)))

                    # EKF 预测 + 更新（自动拒绝跳变 yaw）
                    dt = (time.time() - t_loop) if ekf.initialized else 0.1
                    ekf.predict(dt)
                    if not ekf.initialized:
                        ekf.init_state(fx, fy, fused_yaw, v=0.0)
                    else:
                        ekf.update([fx, fy, fused_yaw])
                    sx, sy, syaw, sv = ekf.get_state()

                    smooth = np.array([sx, sy])

                    # 用 EKF 世界坐标+航向更新 ROI
                    update_shared_roi((float(sx), float(sy)), syaw, cam_params)

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
                        "yaw": round(float(syaw), 4),
                        "n_cams": len(frames),
                        "n_obs": fused["n_obs"],
                        "raw": all_results,
                    }

                    # 发送
                    if not send_result(sock, data):
                        print("[TCP] 连接断开，重连中...")
                        sock.close()
                        sock = connect_to_pc(args.pc_ip, args.port)

                    # 本地打印
                    frame_count += 1
                    if frame_count % 5 == 0:
                        tags_str = " ".join(f"{r['camera']}T{r['tag_id']}" for r in all_results)
                        print(f"\r  XY=({smooth[0]:.3f},{smooth[1]:.3f})  "
                              f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                              f"FPS={fps:.1f}  [{len(frames)}cam, {fused['n_obs']}obs: {tags_str}]  ",
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
