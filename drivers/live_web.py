#!/usr/bin/env python3
"""
PC 端 Web 统一控制台
====================
- 实时追踪显示（开关控制，关则保存日志）
- BEV 俯视图融合（按键触发 → Pi抓图 → PC融合 → 显示）
- 外参自标定（按键触发 → Pi抓图 → Tag PnP标定 → 更新外参）
用法: python drivers/live_web.py
"""

import socket
import json
import sys
import os
import glob as _glob
import threading
import time
import shutil
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"
PI_DIR = "/home/pi/uwb_tracker"
PI_IP = "100.127.223.76"  # PC Tailscale IP

# ---- 全局状态 ----
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {"x": -99, "y": -99, "yaw": 0, "fps": 0, "n_cams": 0, "n_obs": 0,
                      "grid_x": -99, "grid_y": -99, "err_cm": -99}
        self.trail = deque(maxlen=500)
        self.raw_tags = []
        self.tracking_active = True
        self.bev_image = None          # 当前 BEV 图路径
        self.cmd_lock = threading.Lock()  # 命令互斥锁
        self.bev_version = 0
        self.status_msg = ""

state = State()
_last_calib_info = []  # 最近一次标定的相机信息


# ---- SSH 辅助 ----
def _ssh(cmd, timeout=15):
    """执行 SSH 命令，返回 stdout。"""
    import paramiko
    print(f"  [SSH] 执行: {cmd[:80]}...", flush=True)
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode()
        err = stderr.read().decode()
        ssh.close()
        if err.strip():
            print(f"  [SSH] stderr: {err[:200]}", flush=True)
        if out.strip():
            print(f"  [SSH] stdout: {out[:200]}", flush=True)
        return out
    except Exception as e:
        print(f"  [SSH] 失败: {e}", flush=True)
        return ""


def _ssh_upload_file(local, remote):
    """上传本地文件到 Pi。"""
    import paramiko
    print(f"  [SFTP] 上传 {local} -> {remote}", flush=True)
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        sftp = ssh.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        ssh.close()
        return True
    except Exception as e:
        print(f"  [SFTP] 上传失败: {e}", flush=True)
        return False


def _ssh_upload_str(content, remote):
    """上传字符串内容到 Pi 文件。"""
    import paramiko
    print(f"  [SFTP] 上传脚本 -> {remote}", flush=True)
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        sftp = ssh.open_sftp()
        with sftp.file(remote, "w") as f:
            f.write(content)
        sftp.close()
        ssh.close()
        print(f"  [SFTP] 上传完成", flush=True)
        return True
    except Exception as e:
        print(f"  [SFTP] 失败: {e}", flush=True)
        return False


def _ssh_download(remote, local):
    import paramiko
    print(f"  [SFTP] 下载 {remote} -> {local}", flush=True)
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        sftp = ssh.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        ssh.close()
        print(f"  [SFTP] 下载完成", flush=True)
        return True
    except Exception as e:
        print(f"  [SFTP] 下载失败: {e}", flush=True)
        return False


def _stop_tracker():
    print("  [TRACKER] 停止 Pi 追踪...", flush=True)
    _ssh("pkill -9 -f pi_tracker.py 2>/dev/null; sleep 1.5", timeout=10)
    print("  [TRACKER] 已停止", flush=True)


def _start_tracker():
    print(f"  [TRACKER] 启动 Pi 追踪 -> {PI_IP}:9527", flush=True)
    cmd = f"cd {PI_DIR} && nohup python3 pi_tracker.py --pc-ip {PI_IP} --port 9527 > /tmp/pi_tracker.log 2>&1 &"
    _ssh(cmd, timeout=8)
    print("  [TRACKER] 已启动", flush=True)


def _pi_capture_images():
    """Pi 上捕获 3 台相机各一帧，保存到 /tmp/。"""
    print("  [CAP] 开始抓图流程...", flush=True)
    _stop_tracker()
    time.sleep(1.5)

    # 上传抓图脚本
    cap_script = '''import cv2, time, os
cameras = [("/dev/video0", "usb1"), ("/dev/video2", "usb2"), ("/dev/video4", "usb3")]
for dev, name in cameras:
    if not os.path.exists(dev):
        print(name + ":FAIL no device")
        continue
    cap = None
    for attempt in range(5):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        time.sleep(0.8)
    if cap is None or not cap.isOpened():
        print(name + ":FAIL open after 5 tries")
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_GAIN, 16)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 28)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    
    time.sleep(0.3)
    for _ in range(8):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret and frame.mean() > 5:
        path = "/tmp/cap_" + name + ".jpg"
        cv2.imwrite(path, frame)
        print(name + ":OK:" + path + " " + str(frame.shape[1]) + "x" + str(frame.shape[0]))
    else:
        print(name + ":FAIL ret=" + str(ret))
'''
    _ssh_upload_str(cap_script, "/tmp/cap_all.py")
    out = _ssh("python3 /tmp/cap_all.py", timeout=40)
    print(f"  [CAP] Pi 输出: {out.strip()}", flush=True)

    images = {}
    for line in out.strip().split("\n"):
        if ":OK:" in line:
            parts = line.split(":OK:")
            name = parts[0].strip()
            path = parts[1].strip().split()[0]
            local = os.path.join(ROOT, f"tmp_{name}.jpg")
            if _ssh_download(path, local):
                img = cv2.imread(local)
                if img is not None:
                    images[name] = img
                    print(f"  [CAP] {name}: {img.shape}", flush=True)
                else:
                    print(f"  [CAP] {name}: cv2.imread 失败", flush=True)
    print(f"  [CAP] 共捕获 {len(images)} 台相机", flush=True)
    return images


def _do_bev():
    """执行 BEV 融合：先自动标定外参，再生成俯视图。"""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    print("=" * 40, flush=True)
    print("[BEV] 开始 BEV 流程 (标定+融合)", flush=True)
    with state.cmd_lock:
        # 1. 抓图
        state.status_msg = "BEV: 抓图中..."
        images = _pi_capture_images()
        if len(images) < 1:
            _start_tracker()
            state.status_msg = "BEV 失败: 无图像"
            return None

        # 2. 外参自标定
        state.status_msg = f"BEV: 标定外参 ({len(images)}台)..."
        print(f"[BEV] 外参标定 {list(images.keys())}", flush=True)
        _calibrate_from_images(images)

        # 3. BEV 融合
        state.status_msg = f"BEV: 融合俯视图..."
        print(f"[BEV] 融合 {list(images.keys())}", flush=True)
        from src.bev_engine import BevGenerator
        gen = BevGenerator(x_max=5.0, y_min=0.0)
        try:
            fused, tag_data, cam_stats, per_cam_bevs, _masks = gen.run(
                camera_names=list(images.keys()), images=images)
            print(f"[BEV] 融合完成, tag_data={len(tag_data)}", flush=True)
        except Exception as e:
            print(f"[BEV] 融合异常: {e}", flush=True)
            import traceback
            traceback.print_exc()
            _start_tracker()
            state.status_msg = f"BEV 失败: {e}"
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        active = list(images.keys())

        # 保存融合 BEV
        path = os.path.join(ROOT, "bev_latest.jpg")
        cv2.imwrite(path, fused)
        state.bev_version += 1

        # 保存各相机原图 + 单独 BEV
        for name in active:
            cv2.imwrite(os.path.join(ROOT, f"bev_{ts}_{name}_raw.jpg"), images[name])
            cv2.imwrite(os.path.join(ROOT, f"bev_{ts}_{name}_thumb.jpg"),
                        cv2.resize(images[name], (640, 360)))
            if per_cam_bevs.get(name) is not None:
                cv2.imwrite(os.path.join(ROOT, f"bev_{ts}_{name}_bev.jpg"), per_cam_bevs[name])

        # 生成完整 HTML 报告
        _gen_full_report(fused, tag_data, cam_stats, active, ts)
        print(f"[BEV] 报告已保存", flush=True)

        _start_tracker()
        state.status_msg = f"BEV 完成 ({len(images)}cam, {len(tag_data)}tags)"
        print("[BEV] 完成, 追踪已恢复", flush=True)
        return path


def _gen_full_report(fused, tag_data, cam_stats, active, ts):
    """生成完整报告：融合 BEV + 各相机原图 + 单独 BEV。"""
    from collections import Counter

    gen = None
    try:
        from src.bev_engine import BevGenerator
        gen = BevGenerator()
    except Exception:
        pass

    def label(n):
        if gen:
            return gen._name_to_label(n)
        return n

    def color_hex(bgr):
        return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

    # 读取相机参数获取颜色
    params = {}
    try:
        import yaml
        with open("cfg/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for cc in cfg["cameras"]:
            if cc["name"] in active:
                h = __import__("hashlib").md5(cc["name"].encode()).hexdigest()
                params[cc["name"]] = {
                    "color": (int(h[0:2], 16) % 200 + 55, int(h[2:4], 16) % 200 + 55, int(h[4:6], 16) % 200 + 55)
                }
    except Exception:
        for n in active:
            params[n] = {"color": (100, 180, 255)}

    best_counter = Counter(t["best_camera"] for t in tag_data)

    # Tag 表格
    rows = ""
    for t in tag_data:
        gsd_cells = "".join(f"<td>{t['gsd_by_cam'].get(n, '-')}</td>" for n in active)
        best_n = t["best_camera"]
        c = params.get(best_n, {}).get("color", (200, 200, 200))
        hex_c = color_hex(c)
        rows += f'<tr><td>{t["id"]}</td><td>{t["x"]:.2f}</td><td>{t["y"]:.2f}</td><td>{t["n_visible"]}</td>{gsd_cells}<td style="color:{hex_c};font-weight:bold">{label(best_n)}</td></tr>\n'

    gsd_headers = "".join(f"<th>{label(n)} GSD</th>" for n in active)

    # 统计卡片
    cards = ""
    for n in active:
        c = params.get(n, {}).get("color", (200, 200, 200))
        hex_c = color_hex(c)
        cov = cam_stats.get(n, {}).get("coverage_pct", 0)
        cards += f'<div class="card"><h3>{label(n)} Coverage</h3><div class="v" style="color:{hex_c}">{cov}%</div></div>\n'
        cards += f'<div class="card"><h3>{label(n)} Best GSD</h3><div class="v" style="color:{hex_c}">{best_counter.get(n, 0)}</div></div>\n'

    # 各相机原图 + BEV + 标定信息
    calib_info = _last_calib_info or []
    calib_by_name = {}
    for entry in calib_info:
        parts = entry.split(":", 1)
        if len(parts) == 2:
            calib_by_name[parts[0].strip()] = parts[1].strip()

    per_cam_html = ""
    for name in active:
        c = params.get(name, {}).get("color", (200, 200, 200))
        cov = cam_stats.get(name, {}).get("coverage_pct", 0)
        hex_c = color_hex(c)
        cinfo = calib_by_name.get(name, "")
        per_cam_html += f'<div style="flex:1;min-width:300px">\n'
        per_cam_html += f'<p style="color:rgb({c[2]},{c[1]},{c[0]});font-weight:bold">● {label(name)}</p>\n'
        if cinfo:
            per_cam_html += f'<p style="font-size:10px;color:#aac">📐 {cinfo}</p>\n'
        per_cam_html += f'<p style="font-size:11px;color:#888">原图</p>\n'
        per_cam_html += f'<a href="bev_{ts}_{name}_raw.jpg" target="_blank"><img src="bev_{ts}_{name}_thumb.jpg" style="width:100%;border:1px solid #333"></a>\n'
        per_cam_html += f'<p style="font-size:11px;color:#888">BEV ({cov:.0f}%)</p>\n'
        per_cam_html += f'<img src="bev_{ts}_{name}_bev.jpg" style="width:100%;border:1px solid #333">\n'
        per_cam_html += f'</div>\n'

    legend = "".join(
        f'<span style="color:rgb({params[n]["color"][2]},{params[n]["color"][1]},{params[n]["color"][0]})">● {label(n)}</span> '
        for n in active)

    n_cams = len(active)
    n_overlap_all = sum(1 for t in tag_data if t["n_visible"] == n_cams)
    n_overlap_2 = sum(1 for t in tag_data if t["n_visible"] >= 2)
    total = fused.shape[0] * fused.shape[1]
    lit = (fused.sum(axis=2) > 0).sum()

    html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>{n_cams}-Camera BEV Report</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#e94560;border-bottom:2px solid #e94560;padding-bottom:8px}}
h2{{background:#16213e;padding:8px 16px;border-left:3px solid #e94560}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#16213e;border:1px solid #2a2a4a;border-radius:8px;padding:12px 18px;min-width:100px}}
.card h3{{margin:0 0 4px;font-size:11px;color:#888}}
.card .v{{font-size:22px;font-weight:bold}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}}
th{{background:#0f3460;padding:6px 8px;text-align:left;position:sticky;top:0}}
td{{padding:4px 8px;border-bottom:1px solid #2a2a4a}}
tr:nth-child(even){{background:#1e1e35}}
img{{max-width:100%;border:1px solid #2a2a4a;border-radius:4px;margin:8px 0}}
.foot{{color:#666;font-size:11px;margin-top:20px;border-top:1px solid #2a2a4a;padding-top:12px}}
.explain{{background:#1a2a1a;border:1px solid #2a4a2a;border-radius:6px;padding:14px 20px;margin:16px 0;font-size:13px;line-height:1.8}}
.explain strong{{color:#55efc4}}
</style></head><body>
<h1>{n_cams}-Camera BEV Analysis</h1>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
<strong>Cameras:</strong> {', '.join(label(n) for n in active)}</p>

<div class="cards">
<div class="card"><h3>Floor Tags</h3><div class="v">{len(tag_data)}</div></div>
<div class="card"><h3>{n_cams}-Cam Overlap</h3><div class="v">{n_overlap_all}</div></div>
<div class="card"><h3>2-Cam+ Overlap</h3><div class="v">{n_overlap_2}</div></div>
{cards}
</div>

<h2>融合 BEV ({100*lit/total:.0f}% coverage)</h2>
<img src="bev_latest.jpg" alt="Fused BEV">
<p>Tag dots: {legend} Larger dot = more cameras.</p>

<h2>各相机原图 + BEV</h2>
<div style="display:flex;gap:10px;flex-wrap:wrap;">
{per_cam_html}
</div>

<h2>Tag GSD (mm/px) — smaller = better</h2>
<table>
<tr><th>ID</th><th>X</th><th>Y</th><th>#Cam</th>{gsd_headers}<th>Best</th></tr>
{rows}
</table>
<div class="foot">Generated by live_web.py</div>
</body></html>'''

    report_path = os.path.join(ROOT, "bev_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[BEV] 完整报告已保存", flush=True)


def _do_calib_only():
    """单独的外参标定（不生成 BEV）。"""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    print("[CALIB] 开始外参标定", flush=True)
    with state.cmd_lock:
        state.status_msg = "标定: 抓图中..."
        images = _pi_capture_images()
        if len(images) < 1:
            _start_tracker()
            state.status_msg = "标定失败: 无图像"
            return
        _calibrate_from_images(images)
        _start_tracker()
        state.status_msg = "标定完成"


def _calibrate_from_images(images):
    """从捕获的图像中自动标定外参 (AprilTag PnP)，更新 cfg/extrinsics.yaml。"""
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("cfg/floor_tags.yaml", "r", encoding="utf-8") as f:
        ft = yaml.safe_load(f)
    floor_tags = {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
                  for k, v in ft["tags"].items()}
    CART_TAGS = {0, 1, 2, 3}

    from pupil_apriltags import Detector
    detector = Detector(families="tag36h11", quad_decimate=1.0)

    new_ext = {}
    results = []

    for name, img in images.items():
        cc = next(c for c in cfg["cameras"] if c["name"] == name)
        cm = cc["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(cc["dist_coeffs"], dtype=np.float64)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = 0.5
        gray_s = cv2.resize(gray, None, fx=scale, fy=scale)
        gray_s = cv2.createCLAHE(2.0, (8, 8)).apply(gray_s)
        dets = detector.detect(gray_s)
        for d in dets:
            d.corners /= scale
            d.center = (d.center[0] / scale, d.center[1] / scale)

        good = [d for d in dets if d.tag_id in floor_tags and d.tag_id not in CART_TAGS]
        if len(good) < 4:
            results.append(f"{name}: Tag不足({len(good)})")
            continue

        obj_pts, img_pts = [], []
        half = 0.045
        for d in good:
            wpt = floor_tags[d.tag_id]
            c3 = np.array([[wpt[0]-half,wpt[1]-half,0],[wpt[0]+half,wpt[1]-half,0],
                           [wpt[0]+half,wpt[1]+half,0],[wpt[0]-half,wpt[1]+half,0]], dtype=np.float64)
            for ci, ii in zip(c3, d.corners):
                obj_pts.append(ci); img_pts.append(ii)

        ok, rv, tv, inl = cv2.solvePnPRansac(
            np.array(obj_pts, dtype=np.float64), np.array(img_pts, dtype=np.float64),
            K, dist, reprojectionError=8.0, confidence=0.99, iterationsCount=2000)
        if not ok:
            results.append(f"{name}: PnP失败")
            continue

        R, _ = cv2.Rodrigues(rv)
        new_ext[name] = {"R": R.tolist(), "t": tv.flatten().tolist()}
        n_in = len(inl) if inl is not None else 0
        pos = (-R.T @ tv).flatten()
        results.append(f"{name}: {len(good)}tags inliers={n_in} H={abs(pos[2]):.2f}m")

    if new_ext:
        # 合并已有外参（保留未标定的相机）
        try:
            with open("cfg/extrinsics.yaml", "r", encoding="utf-8") as f:
                old_ext = yaml.safe_load(f) or {}
        except Exception:
            old_ext = {}
        old_ext.update(new_ext)
        with open("cfg/extrinsics.yaml", "w", encoding="utf-8") as f:
            yaml.dump(old_ext, f, default_flow_style=None)
        _ssh_upload_file("cfg/extrinsics.yaml", f"{PI_DIR}/cfg/extrinsics.yaml")
        # 保存相机位置信息供报告使用
        global _last_calib_info
        _last_calib_info = results
        print(f"[BEV] 标定: {'; '.join(results)}", flush=True)
    else:
        print(f"[BEV] 标定失败: {'; '.join(results)}，使用旧外参", flush=True)


def _do_calib():
    """执行外参自标定（AprilTag PnP）。"""
    with state.cmd_lock:
        state.status_msg = "标定: 抓图中..."
        images = _pi_capture_images()
        if len(images) < 2:
            _start_tracker()
            state.status_msg = "标定失败: 图像不足"
            return None

        state.status_msg = "标定: 检测 Tag..."

        # 加载配置
        with open("cfg/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        with open("cfg/floor_tags.yaml", "r") as f:
            ft = yaml.safe_load(f)
        floor_tags = {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
                      for k, v in ft["tags"].items()}
        CART_TAGS = {0, 1, 2, 3}

        from pupil_apriltags import Detector
        detector = Detector(families="tag36h11", quad_decimate=1.0)

        new_ext = {}
        results = []

        for name, img in images.items():
            cc = next(c for c in cfg["cameras"] if c["name"] == name)
            cm = cc["camera_matrix"]
            K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
            dist = np.array(cc["dist_coeffs"], dtype=np.float64)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = 0.5
            gray_s = cv2.resize(gray, None, fx=scale, fy=scale)
            gray_s = cv2.createCLAHE(2.0, (8, 8)).apply(gray_s)
            dets = detector.detect(gray_s)
            for d in dets:
                d.corners /= scale
                d.center = (d.center[0] / scale, d.center[1] / scale)

            good = [d for d in dets if d.tag_id in floor_tags and d.tag_id not in CART_TAGS]
            if len(good) < 4:
                results.append(f"{name}: Tag不足({len(good)})")
                continue

            obj_pts, img_pts = [], []
            for d in good:
                wpt = floor_tags[d.tag_id]
                c3 = np.array([[wpt[0]-0.045,wpt[1]-0.045,0],[wpt[0]+0.045,wpt[1]-0.045,0],
                               [wpt[0]+0.045,wpt[1]+0.045,0],[wpt[0]-0.045,wpt[1]+0.045,0]], dtype=np.float64)
                for ci, ii in zip(c3, d.corners):
                    obj_pts.append(ci); img_pts.append(ii)

            ok, rv, tv, inl = cv2.solvePnPRansac(
                np.array(obj_pts, dtype=np.float64), np.array(img_pts, dtype=np.float64),
                K, dist, reprojectionError=8.0, confidence=0.99, iterationsCount=2000)
            if not ok:
                results.append(f"{name}: PnP失败")
                continue

            R, _ = cv2.Rodrigues(rv)
            new_ext[name] = {"R": R.tolist(), "t": tv.flatten().tolist()}
            n_in = len(inl) if inl is not None else 0
            pos = (-R.T @ tv).flatten()
            results.append(f"{name}: {len(good)}tags inliers={n_in} H={abs(pos[2]):.2f}m")

        if new_ext:
            with open("cfg/extrinsics.yaml", "w") as f:
                yaml.dump(new_ext, f, default_flow_style=None)
            # 上传到 Pi
            _ssh_upload_file("cfg/extrinsics.yaml", f"{PI_DIR}/cfg/extrinsics.yaml")
            msg = f"标定完成: {'; '.join(results)}"
        else:
            msg = f"标定失败: {'; '.join(results)}"

        _start_tracker()
        state.status_msg = msg
        return new_ext


# ---- TCP 接收线程 ----
def tcp_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 9527))
    server.listen(1)
    server.settimeout(1.0)
    print("TCP :9527 等待 Pi...", flush=True)

    while True:
        try:
            client, addr = server.accept()
        except socket.timeout:
            continue
        print(f"Pi {addr[0]} 已连接", flush=True)
        buf = ""
        try:
            while True:
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    continue
                except Exception:
                    break
                if not chunk:
                    break
                buf += chunk.decode("utf-8")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.strip():
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        with state.lock:
                            state.data = d
                            if state.tracking_active:
                                x, y = d.get("x", -99), d.get("y", -99)
                                if x != -99:
                                    state.trail.append((x, y))
                                state.raw_tags = d.get("raw", [])
        except Exception:
            pass
        finally:
            client.close()
            print("Pi 断开，等待重连...", flush=True)


# ---- Web UI HTML ----
# ---- Web UI HTML ----
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Car Tracker</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a16;color:#c8c8d8;font-family:'Segoe UI','Consolas','Microsoft YaHei',sans-serif;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#topbar{display:flex;gap:8px;padding:10px 14px;background:#111128;border-bottom:1px solid #282850;
        align-items:center;flex-shrink:0;z-index:10}
#topbar button{padding:8px 16px;border-radius:6px;border:1px solid #383868;background:#1e1e48;
               color:#b0b0d0;font-size:12px;cursor:pointer;transition:.2s;font-family:inherit;white-space:nowrap}
#topbar button:hover{background:#2a2a60;color:#fff;border-color:#5050a0}
#topbar button.on{background:#0a2a40;color:#50b8ff;border-color:#3070a0}
#topbar button.recording{background:#2a0a1a;color:#ff5050;border-color:#803030;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
#topbar .sep{width:1px;height:24px;background:#383868;margin:0 4px}
#topbar .spacer{flex:1}
#topbar .status-msg{font-size:11px;color:#80c080;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#main{flex:1;display:flex;overflow:hidden}
#panel{width:210px;background:#111128;padding:14px;display:flex;flex-direction:column;gap:8px;
       border-right:1px solid #282850;flex-shrink:0;z-index:10}
#panel h3{color:#e94560;font-size:14px;text-align:center;letter-spacing:1px}
#panel .kv{display:flex;justify-content:space-between;font-size:11px}
#panel .kv .k{color:#7878a0}
#panel .kv .v{color:#e8e8f8;font-weight:600;font-family:Consolas,monospace}
#panel .sep{height:1px;background:linear-gradient(90deg,transparent,#3a3a60,transparent)}
#canvas-wrap{flex:1;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center}
#canvas-wrap canvas{position:absolute}
#c-bg{z-index:1}
#c-overlay{z-index:2}
#log-area{position:absolute;bottom:8px;left:14px;right:14px;font-size:10px;color:#606080;z-index:3;
          display:flex;justify-content:space-between;pointer-events:none}
</style>
</head>
<body>
<div id="topbar">
  <button id="btn-track" class="recording" onclick="toggleTrack()">&#x23FA; 追踪中</button>
  <span class="sep"></span>
  <button id="btn-bev" onclick="cmdBev()">&#x1F4F7; BEV 俯视图</button>
  <button id="btn-clear" onclick="clearBev()">&#x2715; 清除背景</button>
  <span class="sep"></span>
  <a href="/report" target="_blank" id="btn-report" style="padding:8px 16px;border-radius:6px;border:1px solid #383868;background:#1e1e48;color:#b0b0d0;font-size:12px;cursor:pointer;text-decoration:none;font-family:inherit;white-space:nowrap" onmouseover="this.style.background='#2a2a60'" onmouseout="this.style.background='#1e1e48'">&#x1F4CA; 标定报告</a>
  <span class="sep"></span>
  <button id="btn-calib" onclick="cmdCalib()">&#x1F527; 外参标定</button>
  <span class="spacer"></span>
  <span class="status-msg" id="status-msg"></span>
</div>
<div id="main">
<div id="panel">
  <h3>CAR TRACKER</h3>
  <div class="sep"></div>
  <div class="kv"><span class="k">XY</span><span class="v" id="xy">--</span></div>
  <div class="kv"><span class="k">Heading</span><span class="v" id="hdg">--</span></div>
  <div class="kv"><span class="k">Grid</span><span class="v" id="grid">--</span></div>
  <div class="kv"><span class="k">Error</span><span class="v" id="err">--</span></div>
  <div class="sep"></div>
  <div class="kv"><span class="k">FPS</span><span class="v" id="fps">--</span></div>
  <div class="kv"><span class="k">Cameras</span><span class="v" id="cam">--</span></div>
  <div class="kv"><span class="k">Tags</span><span class="v" id="obs">--</span></div>
</div>
<div id="canvas-wrap">
  <canvas id="c-bg"></canvas>
  <canvas id="c-overlay"></canvas>
  <div id="log-area"><span id="log-info"></span><span id="trail-count"></span></div>
</div>
</div>
<script>
const XMIN=0,XMAX=5.0,YMIN=0,YMAX=5.0,STEP=0.5;
const BEV_XMIN=0,BEV_XMAX=5.0,BEV_YMIN=0,BEV_YMAX=5.0,BEV_PPM=200,BEV_MARGIN=50;
const BEV_W=(BEV_XMAX-BEV_XMIN)*BEV_PPM+2*BEV_MARGIN;
const BEV_H=(BEV_YMAX-BEV_YMIN)*BEV_PPM+2*BEV_MARGIN;
const cBg=document.getElementById('c-bg'),ctxBg=cBg.getContext('2d');
const c=document.getElementById('c-overlay'),ctx=c.getContext('2d');
let carX=-99,carY=-99,carYaw=0,trail=[],rawTags=[],lastBevVer=-1;
// pre-render arrow as image
const arrowImg=new Image();arrowImg.src="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 40 40%27%3E%3Cdefs%3E%3Cfilter id=%27g%27%3E%3CfeDropShadow dx=%270%27 dy=%271%27 stdDeviation=%271%27 flood-color=%27%23000%27 flood-opacity=%27.4%27/%3E%3C/filter%3E%3C/defs%3E%3Cpath d=%27M38 20 L14 4 L20 16 L4 16 L4 24 L20 24 L14 36Z%27 fill=%27%2340b0ff%27 stroke=%27white%27 stroke-width=%271.5%27 filter=%27url(%23g)%27/%3E%3C/svg%3E";
let bevImg=null,bevOn=false,tracking=true;

function resizeAll(){
  let wrap=c.parentElement, pad=40;
  let maxH=wrap.clientHeight-pad, maxW=wrap.clientWidth-pad;
  let h=Math.floor(Math.min(maxH, maxW*(YMAX-YMIN)/(XMAX-XMIN)));
  let w=Math.floor(h*(XMAX-XMIN)/(YMAX-YMIN));
  cBg.width=w;cBg.height=h;
  c.width=w;c.height=h;
  c.style.left=(wrap.clientWidth-w)/2+"px";
  c.style.top=(wrap.clientHeight-h)/2+"px";
  cBg.style.left=c.style.left;cBg.style.top=c.style.top;
  drawBg();drawOverlay();
}
window.onresize=resizeAll;window.onload=resizeAll;

function ppm(){return c.width/(XMAX-XMIN)}
function w2p(x,y){let p=ppm();return[(x-XMIN)*p,c.height-(y-YMIN)*p]}

// ---- Background layer (BEV, never cleared by tracking) ----
function drawBg(){
  ctxBg.clearRect(0,0,cBg.width,cBg.height);
  if(bevOn&&bevImg){
    let sx=BEV_MARGIN+(XMIN-BEV_XMIN)*BEV_PPM;
    let sy=BEV_H-BEV_MARGIN-(YMAX-BEV_YMIN)*BEV_PPM;
    let sw=(XMAX-XMIN)*BEV_PPM,sh=(YMAX-YMIN)*BEV_PPM;
    ctxBg.drawImage(bevImg,sx,sy,sw,sh,0,0,cBg.width,cBg.height);
  } else {
    let bg=ctxBg.createLinearGradient(0,0,0,cBg.height);
    bg.addColorStop(0,'#0e0e1e');bg.addColorStop(1,'#080812');
    ctxBg.fillStyle=bg;ctxBg.fillRect(0,0,cBg.width,cBg.height);
  }
}

// ---- Overlay layer (grid, trail, car, tags) ----
function drawOverlay(){
  ctx.clearRect(0,0,c.width,c.height);
  let p=ppm();

  // grid
  ctx.strokeStyle='rgba(50,50,80,0.5)';ctx.lineWidth=0.5;
  for(let x=XMIN;x<=XMAX+0.001;x+=STEP){let[u,v]=w2p(x,YMIN),[,v2]=w2p(x,YMAX);
   ctx.beginPath();ctx.moveTo(u,v);ctx.lineTo(u,v2);ctx.stroke()}
  for(let y=YMIN;y<=YMAX+0.001;y+=STEP){let[u,v]=w2p(XMIN,y),[u2]=w2p(XMAX,y);
   ctx.beginPath();ctx.moveTo(u,v);ctx.lineTo(u2,v);ctx.stroke()}
  ctx.fillStyle='#8888aa';ctx.font='bold 11px Consolas';
  for(let x=XMIN;x<=XMAX+0.001;x+=STEP){let[u,]=w2p(x,YMIN);ctx.fillText(x.toFixed(1),u-10,c.height-6)}
  for(let y=YMIN;y<=YMAX+0.001;y+=STEP){let[,v]=w2p(XMIN,y);ctx.fillText(y.toFixed(1),4,v+10)}
  let[u0,v0]=w2p(XMIN,YMIN),[u1,v1]=w2p(XMAX,YMAX);
  ctx.strokeStyle='#3c3c6a';ctx.lineWidth=1.5;ctx.strokeRect(u0,v1,u1-u0,v0-v1);
  let[ox,oy]=w2p(0,0);
  ctx.strokeStyle='rgba(255,255,255,0.12)';
  for(let x=0;x<=XMAX;x+=1){let[u,]=w2p(x,0);ctx.beginPath();ctx.moveTo(u,oy);ctx.lineTo(u,oy-6);ctx.stroke()}
  for(let y=0;y<=YMAX;y+=1){let[,v]=w2p(0,y);ctx.beginPath();ctx.moveTo(ox,v);ctx.lineTo(ox+6,v);ctx.stroke()}

  // raw tags
  for(let t of rawTags){let[u,v]=w2p(t.center_xy[0],t.center_xy[1]);
   ctx.fillStyle='rgba(80,255,120,0.3)';ctx.beginPath();ctx.arc(u,v,2.5,0,Math.PI*2);ctx.fill()}

  // trail
  if(trail.length>1){
   ctx.strokeStyle='rgba(0,220,240,0.5)';ctx.lineWidth=3;ctx.lineCap='round';
   ctx.beginPath();let[f]=w2p(trail[0][0],trail[0][1]);ctx.moveTo(f[0],f[1]);
   for(let i=1;i<trail.length;i++){let[u,v]=w2p(trail[i][0],trail[i][1]);ctx.lineTo(u,v)}ctx.stroke()}

  // car
  if(carX!=-99){
   let[u,v]=w2p(carX,carY);
   // glow
   let g=ctx.createRadialGradient(u,v,0,u,v,14);
   g.addColorStop(0,'rgba(255,180,30,0.3)');g.addColorStop(1,'rgba(255,180,30,0)');
   ctx.fillStyle=g;ctx.beginPath();ctx.arc(u,v,14,0,Math.PI*2);ctx.fill();
   // rear dot
   ctx.fillStyle='#f0503c';ctx.beginPath();ctx.arc(u,v,6,0,Math.PI*2);ctx.fill();
   ctx.strokeStyle='rgba(0,0,0,0.4)';ctx.lineWidth=1.5;ctx.stroke();
   // arrow image
   if(arrowImg.complete){
    ctx.save();ctx.translate(u,v);ctx.rotate(-carYaw);
    ctx.drawImage(arrowImg,-8,-20,40,40);
    ctx.restore();
   }
   // FRONT
   ctx.fillStyle='#fff';ctx.font='bold 9px Consolas';
   
  }
  document.getElementById('trail-count').textContent=trail.length+' pts';
}


function update(d){
  carX=d.x;carY=d.y;carYaw=d.yaw||0;
  document.getElementById('xy').textContent=carX!=-99?`(${carX.toFixed(3)},${carY.toFixed(3)})`:'--';
  let deg=carYaw*180/Math.PI;let dirs=['N','NE','E','SE','S','SW','W','NW'];let di=Math.round(((deg%360+360)%360)/45)%8;document.getElementById('hdg').textContent=carX!=-99?deg.toFixed(0)+'\u00b0 '+dirs[di]:'--';
  document.getElementById('grid').textContent=carX!=-99?`(${(d.grid_x??-99).toFixed(1)},${(d.grid_y??-99).toFixed(1)})`:'--';
  document.getElementById('err').textContent=carX!=-99?`${(d.err_cm??0).toFixed(1)}cm`:'--';
  document.getElementById('fps').textContent=(d.fps??0).toFixed(1);
  document.getElementById('cam').textContent=`${d.n_cams??0}/3`;
  document.getElementById('obs').textContent=`${d.n_obs??0}`;
  drawOverlay();
}

function toggleTrack(){
  tracking=!tracking;let btn=document.getElementById('btn-track');
  if(tracking){btn.textContent='\u23FA 追踪中';btn.className='recording';trail=[];rawTags=[];document.getElementById('log-info').textContent='';
   fetch('/cmd?action=track_start');}
  else{btn.textContent='\u25B6 已停止';btn.className='';
   let log={time:new Date().toISOString(),trail:trail.map(p=>({x:p[0],y:p[1]}))};
   let logs=JSON.parse(localStorage.getItem('track_logs')||'[]');logs.push(log);
   if(logs.length>20)logs=logs.slice(-20);localStorage.setItem('track_logs',JSON.stringify(logs));
   document.getElementById('log-info').textContent='已保存 '+trail.length+' 点 | 共'+logs.length+'条记录';
   trail=[];rawTags=[];carX=-99;carY=-99;drawOverlay();fetch('/cmd?action=track_stop');}
}

function cmdBev(){document.getElementById('status-msg').textContent='BEV: 处理中...';fetch('/cmd?action=bev').then(r=>r.json());}
function cmdCalib(){document.getElementById('status-msg').textContent='标定: 处理中...';fetch('/cmd?action=calib').then(r=>r.json());}
function clearBev(){bevOn=false;bevImg=null;drawBg();document.getElementById('status-msg').textContent='BEV已清除'}
function loadBevBg(){let img=new Image();img.onload=()=>{bevImg=img;bevOn=true;drawBg()};img.onerror=()=>{bevOn=false};img.src='/bev.jpg?t='+Date.now()}

let _lastStatus='';
function checkStatus(msg){
  if(msg!==_lastStatus){_lastStatus=msg;document.getElementById('status-msg').textContent=msg||''}
}

// SSE
fetch('/stream').then(r=>{
  let reader=r.body.getReader(),decoder=new TextDecoder(),buf='';
  function pump(){reader.read().then(({done,value})=>{
   if(done)return;
   buf+=decoder.decode(value,{stream:true});
   let lines=buf.split('\n');buf=lines.pop();
   for(let l of lines){if(l.startsWith('data:')){
    try{let d=JSON.parse(l.slice(5));
     if(d.status_msg)checkStatus(d.status_msg);
     if(d.bev_version!==undefined&&d.bev_version!==lastBevVer){lastBevVer=d.bev_version;loadBevBg()}
     if(tracking){
      if(d.x!=-99){if(!trail.length||trail[trail.length-1][0]!==d.x||trail[trail.length-1][1]!==d.y)trail.push([d.x,d.y]);rawTags=d.raw||[]}
      else{rawTags=[]}
     }
     update(d);
    }catch(e){}
   }}pump()})}pump()})
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/" or p.path == "/index.html":
            self._serve_html()
        elif p.path == "/bev.jpg":
            self._serve_bev()
        elif p.path == "/report":
            self._serve_report()
        elif p.path == "/stream":
            self._serve_sse()
        elif p.path == "/cmd":
            self._handle_cmd(parse_qs(p.query))
        elif p.path.endswith(".jpg") or p.path.endswith(".png"):
            self._serve_file(p.path)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def _serve_bev(self):
        bev = os.path.join(ROOT, "bev_latest.jpg")
        if not os.path.exists(bev):
            bev_glob = sorted(_glob.glob(os.path.join(ROOT, "bev_*_fused.jpg")), reverse=True)
            if bev_glob:
                bev = bev_glob[0]
        if os.path.exists(bev):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(bev, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_report(self):
        report = os.path.join(ROOT, "bev_report.html")
        if not os.path.exists(report):
            reports = sorted(_glob.glob(os.path.join(ROOT, "bev_*_report.html")), reverse=True)
            if reports:
                report = reports[0]
        if os.path.exists(report):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(report, "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path):
        """服务根目录下的静态文件（报告引用的图片等）。"""
        safe = path.lstrip("/").replace("\\", "/")
        full = os.path.normpath(os.path.join(ROOT, safe))
        if not full.startswith(os.path.normpath(ROOT)):
            self.send_response(403); self.end_headers(); return
        if os.path.isfile(full):
            ct = "image/jpeg" if full.endswith(".jpg") else "image/png"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            with open(full, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_cmd(self, qs):
        action = qs.get("action", [""])[0]
        print(f"[CMD] 收到命令: {action}", flush=True)

        if action == "bev":
            state.status_msg = "BEV: 处理中..."
            # 异步执行，不阻塞 HTTP 响应
            threading.Thread(target=_do_bev, daemon=True).start()
            self._json_reply(True, "BEV: 处理中...")
            return
        elif action == "calib":
            state.status_msg = "标定: 处理中..."
            threading.Thread(target=_do_calib_only, daemon=True).start()
            self._json_reply(True, "标定: 处理中...")
            return
        elif action == "track_start":
            with state.lock:
                state.tracking_active = True
                state.trail.clear()
            self._json_reply(True, "追踪已开始")
            return
        elif action == "track_stop":
            with state.lock:
                state.tracking_active = False
            self._json_reply(True, "追踪已停止")
            return
        else:
            self._json_reply(False, "未知命令")

    def _json_reply(self, ok, msg):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok, "msg": msg}).encode())

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                with state.lock:
                    d = dict(state.data)
                    d["raw"] = list(state.raw_tags) if state.tracking_active else []
                    d["status_msg"] = state.status_msg
                    d["bev_version"] = state.bev_version
                line = "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                try:
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                except Exception:
                    break
                time.sleep(0.05)
        except Exception:
            pass

    def log_message(self, format, *args):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass


def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    import sys
    if "--bev-test" in sys.argv:
        # 直接测试 BEV 流程
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s", stream=sys.stderr)
        print("Testing BEV...", flush=True)
        path = _do_bev()
        print(f"Result: {path}", flush=True)
        return

    t = threading.Thread(target=tcp_receiver, daemon=True)
    t.start()

    httpd = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    local_url = "http://localhost:8080"
    lan_ip = _get_lan_ip()
    lan_url = f"http://{lan_ip}:8080" if lan_ip else None

    print("", flush=True)
    print("  ========================================", flush=True)
    print("  \033]8;;" + local_url + "\033\\" + local_url + "\033]8;;\033\\", flush=True)
    if lan_url:
        print("  \033]8;;" + lan_url + "\033\\" + lan_url + "\033]8;;\033\\  (手机/平板)", flush=True)
    print("  ========================================", flush=True)
    print("  Buttons:", flush=True)
    print("    [Track]  - toggle real-time tracking (off = save log)", flush=True)
    print("    [BEV]    - Pi capture -> PC fuse -> show background", flush=True)
    print("    [Clear]  - remove BEV overlay", flush=True)
    print("    [Calib]  - Pi capture -> Tag PnP auto-calibrate", flush=True)
    print("", flush=True)

    import webbrowser
    webbrowser.open(local_url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n关闭服务...", flush=True)
        _ssh("pkill -f pi_tracker.py 2>/dev/null", timeout=8)
        httpd.shutdown()
        print("Pi 追踪已停止，Web 已关闭", flush=True)


if __name__ == "__main__":
    main()
