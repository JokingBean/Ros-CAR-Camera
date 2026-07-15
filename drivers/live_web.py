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
_log_buffer = []  # 运行时缓冲所有跟踪数据
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
        print(f"  [SSH] 失败: {repr(e)}", flush=True)
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
    """上传最新代码和配置到 Pi 并启动追踪服务。"""
    # 上传 pi_tracker.py
    print(f"  [TRACKER] 上传 pi_tracker.py -> {PI_DIR}/pi_tracker.py", flush=True)
    _ssh_upload_file(os.path.join(ROOT, "pi_tracker.py"), f"{PI_DIR}/pi_tracker.py")

    # 上传配置文件
    for fname in ["config.yaml", "extrinsics.yaml"]:
        local = os.path.join(ROOT, "cfg", fname)
        remote = f"{PI_DIR}/cfg/{fname}"
        print(f"  [TRACKER] 上传 cfg/{fname} -> {remote}", flush=True)
        _ssh_upload_file(local, remote)

    print(f"  [TRACKER] 启动 Pi 追踪 -> {PI_IP}:9527", flush=True)
    cmd = f"cd {PI_DIR} && nohup python3 pi_tracker.py --pc-ip {PI_IP} --port 9527 > /tmp/pi_tracker.log 2>&1 &"
    # 后台启动命令用独立 SSH 连接，不用 _ssh（paramiko exec_command 读已关闭通道会报假错）
    import paramiko
    try:
        _csh = paramiko.SSHClient()
        _csh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _csh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        _csh.exec_command(cmd, timeout=5)
        _csh.close()
    except Exception:
        pass  # nohup & 关闭通道后的异常是正常的，忽略
    print("  [TRACKER] 已启动", flush=True)


def _pi_capture_images():
    """Pi 上捕获 3 台相机各一帧，保存到 /tmp/。"""
    print("  [CAP] 开始抓图流程...", flush=True)
    _stop_tracker()
    time.sleep(1.5)

    # 上传抓图脚本（动态从 cfg/config.yaml 读取设备号）
    with open(os.path.join(ROOT, "cfg", "config.yaml"), "r", encoding="utf-8") as _f_cfg:
        _cfg_data = yaml.safe_load(_f_cfg)
    _cam_list = []
    for _c in _cfg_data["cameras"]:
        _dev = _c.get("device", "0")
        _dev_path = f"/dev/video{_dev}"
        _cam_list.append((_dev_path, _c["name"]))
    _cameras_json = json.dumps(_cam_list)
    cap_script = """import cv2, time, os, subprocess as sp
cameras = """ + _cameras_json + """
for dev, name in cameras:
    if not os.path.exists(dev):
        print(name + ":FAIL no device")
        continue
    # 用 v4l2-ctl 预置硬件参数（OpenCV V4L2 映射不正确）
    sp.run(f"v4l2-ctl -d {dev} --set-ctrl=auto_exposure=3,white_balance_automatic=1".split(), capture_output=True, timeout=5)
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
"""
    _ssh_upload_str(cap_script, "/tmp/cap_all.py")
    out = _ssh("python3 /tmp/cap_all.py", timeout=40)
    print(f"  [CAP] Pi 输出: {out.strip()}", flush=True)

    images = {}
    for line in out.strip().split("\n"):
        if ":OK:" in line:
            parts = line.split(":OK:")
            name = parts[0].strip()
            path = parts[1].strip().split()[0]
            local = os.path.join(ROOT, "tmp_imgs", f"tmp_{name}.jpg")
            os.makedirs(os.path.dirname(local), exist_ok=True)
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
    print("[BEV] 开始 BEV 流程", flush=True)
    with state.cmd_lock:
        # 1. Pi 端执行外参标定
        state.status_msg = "BEV: 标定外参..."
        print("[BEV] Pi 端标定外参...", flush=True)
        calib_results = _pi_calibrate()
        if calib_results:
            for r in calib_results:
                print(f"  {r}", flush=True)

        # 2. 抓图用于 BEV 融合
        state.status_msg = "BEV: 抓图中..."
        images = _pi_capture_images()
        if len(images) < 1:
            _start_tracker()
            state.status_msg = "BEV 失败: 无图像"
            return None

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

        # 创建本次输出目录
        out_dir = os.path.join(ROOT, "bev_output", ts)
        os.makedirs(out_dir, exist_ok=True)

        # 保存融合 BEV
        path = os.path.join(out_dir, "bev_fused.jpg")
        cv2.imwrite(path, fused)
        # 也更新 bev_latest 软链接
        latest_path = os.path.join(ROOT, "bev_latest.jpg")
        cv2.imwrite(latest_path, fused)
        state.bev_version += 1

        # 保存各相机原图 + 单独 BEV
        for name in active:
            cv2.imwrite(os.path.join(out_dir, f"bev_{name}_raw.jpg"), images[name])
            cv2.imwrite(os.path.join(out_dir, f"bev_{name}_thumb.jpg"),
                        cv2.resize(images[name], (640, 360)))
            if per_cam_bevs.get(name) is not None:
                cv2.imwrite(os.path.join(out_dir, f"bev_{name}_bev.jpg"), per_cam_bevs[name])

        # 生成完整 HTML 报告
        _gen_full_report(fused, tag_data, cam_stats, active, ts, out_dir)
        print(f"[BEV] 报告已保存", flush=True)

        _start_tracker()
        state.status_msg = f"BEV 完成 ({len(images)}cam, {len(tag_data)}tags)"
        print("[BEV] 完成, 追踪已恢复", flush=True)
        return path


def _gen_full_report(fused, tag_data, cam_stats, active, ts, out_dir):
    """生成完整报告：融合 BEV + 各相机原图 + 单独 BEV。"""
    from collections import Counter

    web_dir = os.path.relpath(out_dir, ROOT).replace("\\", "/")

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
        per_cam_html += f'<a href="/{web_dir}/bev_{name}_raw.jpg" target="_blank"><img src="/{web_dir}/bev_{name}_thumb.jpg" style="width:100%;border:1px solid #333"></a>\n'
        per_cam_html += f'<p style="font-size:11px;color:#888">BEV ({cov:.0f}%)</p>\n'
        per_cam_html += f'<img src="/{web_dir}/bev_{name}_bev.jpg" style="width:100%;border:1px solid #333">\n'
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
<img src="/{web_dir}/bev_fused.jpg" alt="Fused BEV">
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

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[BEV] 完整报告已保存", flush=True)


def _do_calib_only():
    """单独的外参标定 — Pi 端执行 3 次取中位数。"""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    print("[CALIB] 开始外参标定（Pi 端执行 3 次取中位数）", flush=True)
    with state.cmd_lock:
        state.status_msg = "标定: 第 1/3 次..."

        # 收集 3 次标定结果
        all_ext_runs = []
        all_results_msgs = []
        for run_i in range(3):
            if run_i > 0:
                state.status_msg = f"标定: 第 {run_i+1}/3 次..."
            ext_data = _pi_calibrate()
            # ext_data 是 results 字符串列表
            all_results_msgs.append(ext_data)
            # 读取刚刚保存的外参
            try:
                with open("cfg/extrinsics.yaml", "r") as f:
                    run_ext = yaml.safe_load(f) or {}
                all_ext_runs.append(run_ext)
            except Exception:
                pass

        # 3 次后取中位数
        if len(all_ext_runs) >= 2:
            import copy
            median_ext = {}
            cam_names = all_ext_runs[0].keys()
            for cam in cam_names:
                # 收集该相机在所有 run 中的 R, t
                Rs = [run[cam]["R"] for run in all_ext_runs if cam in run]
                ts = [run[cam]["t"] for run in all_ext_runs if cam in run]
                if len(Rs) >= 2:
                    # 元素级中位数
                    R_med = np.median(np.array(Rs), axis=0).tolist()
                    t_med = np.median(np.array(ts), axis=0).tolist()
                    # 确保 R 是有效旋转矩阵（SVD 正交化）
                    R_arr = np.array(R_med)
                    U, _, Vt = np.linalg.svd(R_arr)
                    R_ortho = (U @ Vt).tolist()
                    median_ext[cam] = {"R": R_ortho, "t": t_med}
                elif Rs:
                    median_ext[cam] = {"R": Rs[0], "t": ts[0]}

            if median_ext:
                # 合并旧外参并保存
                try:
                    with open("cfg/extrinsics.yaml", "r") as f:
                        old_ext = yaml.safe_load(f) or {}
                except Exception:
                    old_ext = {}
                old_ext.update(median_ext)
                with open("cfg/extrinsics.yaml", "w") as f:
                    yaml.dump(old_ext, f, default_flow_style=None)
                print(f"  [CALIB] 中位数外参已保存: {list(median_ext.keys())}", flush=True)

        # 展示最后一次的结果消息
        final_results = all_results_msgs[-1] if all_results_msgs else []
        _start_tracker()
        if final_results:
            msg = "; ".join(final_results)
            print(f"  [CALIB] 结果: {msg}", flush=True)
            state.status_msg = f"标定完成: {msg}"
        else:
            state.status_msg = "标定失败"


def _pi_calibrate():
    """生成标定脚本 → Pi 端执行 → 下载结果 → 返回结果列表。"""
    # 先停追踪释放摄像头
    _stop_tracker()
    time.sleep(1)
    # 读取本地配置，嵌入到 Pi 脚本中
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("cfg/floor_tags.yaml", "r", encoding="utf-8") as f:
        ft = yaml.safe_load(f)

    # 构建 Pi 端标定脚本
    script_lines = [
        "import cv2, numpy as np, json, os, yaml, time",
        "from pupil_apriltags import Detector",
        "",
        "CFG_DIR = '/home/pi/uwb_tracker/cfg'",
        f"floor_tags = {json.dumps(ft['tags'])}",
        "",
        "CART_TAGS = {0, 1, 2, 3}",
        "detector = Detector(families='tag36h11', quad_decimate=1.0)",
        "",
        "new_ext = {}",
        "results = []",
        "",
        "# 相机配置（从 PC 端 config.yaml 读取）",
        f"cameras_cfg = {json.dumps(cfg['cameras'])}",
        "",
        "for cc in cameras_cfg:",
        "    name = cc['name']",
        "    dev_str = cc['device']",
        "    if isinstance(dev_str, str) and 'video' in dev_str: dev = int(dev_str.split('video')[-1])",
        "    elif isinstance(dev_str, str): dev = int(dev_str)",
        "    else: dev = int(dev_str)",
        "    cm = cc['camera_matrix']",
        "    K = np.array([[cm['fx'],0,cm['cx']],[0,cm['fy'],cm['cy']],[0,0,1]], dtype=np.float64)",
        "    dist = np.array(cc['dist_coeffs'], dtype=np.float64)",
        "",
        "    # 打开相机抓图（带重试）",
        "    cap = None",
        "    for attempt in range(5):",
        "        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)",
        "        if cap.isOpened(): break",
        "        time.sleep(0.8)",
        "    if cap is None or not cap.isOpened():",
        "        results.append(f'{name}: 无法打开摄像头')",
        "        continue",
        "    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))",
        "    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)",
        "    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)",
        "    time.sleep(0.3)",
        "    for _ in range(10): cap.read()",
        "    ret, frame = cap.read()",
        "    cap.release()",
        "    if not ret:",
        "        results.append(f'{name}: 抓图失败')",
        "        continue",
        "",
        "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
        "    gray_s = cv2.resize(gray, None, fx=0.5, fy=0.5)",
        "    gray_s = cv2.createCLAHE(2.0, (8,8)).apply(gray_s)",
        "    dets = detector.detect(gray_s)",
        "    for d in dets:",
        "        d.corners /= 0.5",
        "        d.center = (d.center[0]/0.5, d.center[1]/0.5)",
        "",
        "    good = [d for d in dets if str(d.tag_id) in floor_tags and d.tag_id not in CART_TAGS]",
        "    if len(good) < 8:",
        "        results.append(f'{name}: Tag不足({len(good)})')",
        "        continue",
        "",
        "    obj_pts, img_pts = [], []",
        "    for d in good:",
        "        wpt = floor_tags[str(d.tag_id)]",
        "        half = 0.045",
        "        c3 = np.array([[wpt['x']-half,wpt['y']-half,0],[wpt['x']+half,wpt['y']-half,0],",
        "                       [wpt['x']+half,wpt['y']+half,0],[wpt['x']-half,wpt['y']+half,0]], dtype=np.float64)",
        "        for ci, ii in zip(c3, d.corners):",
        "            obj_pts.append(ci); img_pts.append(ii)",
        "",
        "    ok, rv, tv, inl = cv2.solvePnPRansac(",
        "        np.array(obj_pts, dtype=np.float64), np.array(img_pts, dtype=np.float64),",
        "        K, dist, reprojectionError=4.0, confidence=0.99, iterationsCount=2000)",
        "    if not ok:",
        "        results.append(f'{name}: PnP失败')",
        "        continue",
        "",
        "    R, _ = cv2.Rodrigues(rv)",
        "    new_ext[name] = {'R': R.tolist(), 't': tv.flatten().tolist()}",
        "    n_in = len(inl) if inl is not None else 0",
        "    pos = (-R.T @ tv).flatten()",
        "",
        "    # 计算重投影误差",
        "    proj, _ = cv2.projectPoints(np.array(obj_pts), rv, tv, K, dist)",
        "    reproj_err = float(np.mean([np.linalg.norm(proj[i].flatten()-img_pts[i]) for i in range(len(obj_pts))]))",
        "    results.append(f'{name}: {len(good)}tags inliers={n_in} H={abs(pos[2]):.2f}m reproj={reproj_err:.1f}px')",
        "",
        "if new_ext:",
        "    # 合并旧外参",
        "    ext_path = os.path.join(CFG_DIR, 'extrinsics.yaml')",
        "    old_ext = {}",
        "    if os.path.exists(ext_path):",
        "        with open(ext_path) as f:",
        "            old_ext = yaml.safe_load(f) or {}",
        "    old_ext.update(new_ext)",
        "    with open(ext_path, 'w') as f:",
        "        yaml.dump(old_ext, f, default_flow_style=None)",
        "# 始终写出结果供 PC 下载（含失败信息）",
        "with open('/tmp/pi_calib_result.json', 'w') as f:",
        "    json.dump({'ext': new_ext, 'results': results}, f)",
        "# 同时打印 JSON 到 stdout 供 PC 直接解析",
        "import sys as _sys",
        "_sys.stdout.flush()",
        "print('===CALIB_RESULT_START===')",
        "_sys.stdout.flush()",
        "print(json.dumps({'ext': new_ext, 'results': results}))",
        "_sys.stdout.flush()",
        "print('===CALIB_RESULT_END===')",
        "",
        "for r in results:",
        "    print(r)",
        "print('DONE')",
    ]

    script = "\n".join(script_lines)
    _ssh_upload_str(script, "/tmp/pi_calibrate.py")
    out = _ssh("python3 /tmp/pi_calibrate.py", timeout=60)
    print(f"  [CALIB] Pi 输出:\n{out}", flush=True)

    # 下载标定结果（优先从 stdout JSON 解析）
    results = []
    # 从 stdout 提取 JSON 结果
    import re as _re
    m = _re.search(r'===CALIB_RESULT_START===\n(.*?)\n===CALIB_RESULT_END===', out, _re.DOTALL)
    if m:
        try:
            calib_data = json.loads(m.group(1))
            results = calib_data.get("results", [])
            new_ext = calib_data.get("ext", {})
            if new_ext:
                try:
                    with open("cfg/extrinsics.yaml", "r") as f:
                        old_ext = yaml.safe_load(f) or {}
                except Exception:
                    old_ext = {}
                old_ext.update(new_ext)
                with open("cfg/extrinsics.yaml", "w") as f:
                    yaml.dump(old_ext, f, default_flow_style=None)
                print(f"  [CALIB] 外参已更新: {list(new_ext.keys())}", flush=True)
        except Exception as e:
            print(f"  [CALIB] 解析标定结果失败: {e}", flush=True)

    # 回退：如果 stdout 没有 JSON，尝试下载文件
    if not results:
        try:
            import paramiko
            _csh = paramiko.SSHClient()
            _csh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            _csh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
            sftp = _csh.open_sftp()
            sftp.get("/tmp/pi_calib_result.json", "/tmp/pi_calib_result.json")
            sftp.close()
            _csh.close()
            with open("/tmp/pi_calib_result.json", "r") as f:
                calib_data = json.load(f)
            results = calib_data.get("results", [])
            new_ext = calib_data.get("ext", {})
            if new_ext:
                try:
                    with open("cfg/extrinsics.yaml", "r") as f:
                        old_ext = yaml.safe_load(f) or {}
                except Exception:
                    old_ext = {}
                old_ext.update(new_ext)
                with open("cfg/extrinsics.yaml", "w") as f:
                    yaml.dump(old_ext, f, default_flow_style=None)
                print(f"  [CALIB] 外参已更新: {list(new_ext.keys())}", flush=True)
        except Exception as e:
            print(f"  [CALIB] 下载结果失败: {e}", flush=True)

    # 最后回退：从 stdout 文本行解析
    if not results:
        for line in out.strip().split("\n"):
            line = line.strip()
            if line and "tags" in line and ("inliers" in line or "H=" in line):
                results.append(line)

    return results


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
                        # 打印位置数据到控制台
                        if d.get("x", -99) != -99:
                            raw = d.get("raw", [])
                            cam_str = "  ".join(
                                f"{r['camera']}T{r['tag_id']}({r['center_xy'][0]:.3f},{r['center_xy'][1]:.3f})"
                                for r in raw[:6])
                            print(f"  XY=({d['x']:.3f},{d['y']:.3f}) yaw={d.get('yaw',0):.1f}° "
                                  f"obs={d.get('n_obs',0)} [{cam_str}]  "
                                  f"yaw_raw=[{','.join(d.get('raw_yaws',[]))}]  "
                                  f"FPS={d.get('fps',0):.1f}",
                                  end="\n", flush=True)
                            _log_buffer.append({
                                "t": d.get("t", 0),
                                "x": d.get("x", -99), "y": d.get("y", -99),
                                "yaw": d.get("yaw", 0), "fps": d.get("fps", 0),
                                "n_cams": d.get("n_cams", 0), "n_obs": d.get("n_obs", 0),
                            })
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
body{background:#f5f7fa;color:#333;font-family:'Segoe UI','Consolas','Microsoft YaHei',sans-serif;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#topbar{display:flex;gap:8px;padding:10px 14px;background:#fff;border-bottom:1px solid #e0e0e0;
        align-items:center;flex-shrink:0;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,0.06)}
#topbar button{padding:8px 16px;border-radius:6px;border:1px solid #d0d0d0;background:#fff;
               color:#555;font-size:12px;cursor:pointer;transition:.2s;font-family:inherit;white-space:nowrap}
#topbar button:hover{background:#f0f4ff;color:#2563eb;border-color:#2563eb}
#topbar button.on{background:#e8f4ff;color:#2563eb;border-color:#2563eb}
#topbar button.recording{background:#fff0f0;color:#dc2626;border-color:#dc2626;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
#topbar .sep{width:1px;height:24px;background:#e0e0e0;margin:0 4px}
#topbar .spacer{flex:1}
#topbar .status-msg{font-size:11px;color:#059669;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#main{flex:1;display:flex;overflow:hidden}
#panel{width:210px;background:#fff;padding:14px;display:flex;flex-direction:column;gap:8px;
       border-right:1px solid #e0e0e0;flex-shrink:0;z-index:10}
#panel h3{color:#2563eb;font-size:14px;text-align:center;letter-spacing:1px}
#panel .kv{display:flex;justify-content:space-between;font-size:11px}
#panel .kv .k{color:#888}
#panel .kv .v{color:#222;font-weight:600;font-family:Consolas,monospace}
#panel .sep{height:1px;background:linear-gradient(90deg,transparent,#d0d0d0,transparent)}
#canvas-wrap{flex:1;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center}
#canvas-wrap canvas{position:absolute}
#c-bg{z-index:1}
#c-overlay{z-index:2}
#log-area{position:absolute;bottom:8px;left:14px;right:14px;font-size:10px;color:#999;z-index:3;
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
  <a href="/report" target="_blank" id="btn-report" style="padding:8px 16px;border-radius:6px;border:1px solid #d0d0d0;background:#fff;color:#555;font-size:12px;cursor:pointer;text-decoration:none;font-family:inherit;white-space:nowrap" onmouseover="this.style.background='#f0f4ff';this.style.color='#2563eb'" onmouseout="this.style.background='#fff';this.style.color='#555'">&#x1F4CA; 标定报告</a>
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
    bg.addColorStop(0,'#eef2f7');bg.addColorStop(1,'#e4e9f0');
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
   ctx.strokeStyle='rgba(0,0,0,0.55)';ctx.lineWidth=3;ctx.lineCap='round';
   ctx.beginPath();let[f]=w2p(trail[0][0],trail[0][1]);ctx.moveTo(f[0],f[1]);
   for(let i=1;i<trail.length;i++){let[u,v]=w2p(trail[i][0],trail[i][1]);ctx.lineTo(u,v)}ctx.stroke()}

  // car
  if(carX!=-99){
   let[u,v]=w2p(carX,carY);
   // glow
   let g=ctx.createRadialGradient(u,v,0,u,v,14);
   g.addColorStop(0,'rgba(37,99,235,0.2)');g.addColorStop(1,'rgba(37,99,235,0)');
   ctx.fillStyle=g;ctx.beginPath();ctx.arc(u,v,14,0,Math.PI*2);ctx.fill();
   // arrow: car heading direction (with shadow for visibility)
   let aLen=35, aAngle=carYaw, arrColor='#f97316';
   let tip=[u+aLen*Math.cos(-aAngle), v+aLen*Math.sin(-aAngle)];
   // shadow
   ctx.shadowColor='rgba(0,0,0,0.4)';ctx.shadowBlur=6;ctx.shadowOffsetX=1;ctx.shadowOffsetY=1;
   // arrow body
   ctx.strokeStyle=arrColor;ctx.lineWidth=5;ctx.lineCap='round';
   ctx.beginPath();ctx.moveTo(u,v);ctx.lineTo(tip[0],tip[1]);ctx.stroke();
   // arrow head
   let hw=10, hl=14;
   let h1=[tip[0]-hl*Math.cos(-aAngle-0.5), tip[1]-hl*Math.sin(-aAngle-0.5)];
   let h2=[tip[0]-hl*Math.cos(-aAngle+0.5), tip[1]-hl*Math.sin(-aAngle+0.5)];
   ctx.fillStyle=arrColor;
   ctx.beginPath();ctx.moveTo(tip[0],tip[1]);ctx.lineTo(h1[0],h1[1]);ctx.lineTo(h2[0],h2[1]);ctx.closePath();ctx.fill();
   ctx.shadowColor='transparent';ctx.shadowBlur=0;
   // center dot (on top of arrow)
   ctx.fillStyle='#2563eb';ctx.beginPath();ctx.arc(u,v,7,0,Math.PI*2);ctx.fill();
   ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();
   
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

// SSE with auto-reconnect
function connectSSE(){
 fetch('/stream').then(r=>{
  let reader=r.body.getReader(),decoder=new TextDecoder(),buf='';
  function pump(){reader.read().then(({done,value})=>{
   if(done){setTimeout(connectSSE,1000);return}
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
   }}pump()})}pump()
 }).catch(()=>setTimeout(connectSSE,1000))
}connectSSE()
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
            reports = sorted(_glob.glob(os.path.join(ROOT, "bev_output", "*", "report.html")), reverse=True)
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

        # 保存跟踪日志
        if _log_buffer:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(ROOT, f"track_log_{ts}.jsonl")
            with open(log_path, "w") as f:
                for entry in _log_buffer:
                    f.write(json.dumps(entry) + "\n")
            print(f"  跟踪日志已保存: {log_path} ({len(_log_buffer)} frames)", flush=True)

        # 下载 Pi 端日志（含计时信息）
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(PI_HOST, username=PI_USER, password=[redacted], timeout=10)
            sftp = ssh.open_sftp()
            pi_log = os.path.join(ROOT, f"pi_tracker_{ts}.log")
            sftp.get("/tmp/pi_tracker.log", pi_log)
            pi_timing = os.path.join(ROOT, f"pi_timing_{ts}.csv")
            try:
                sftp.get("/tmp/pi_timing.csv", pi_timing)
                print(f"  Pi 计时日志已下载: {pi_timing}", flush=True)
            except:
                pass
            sftp.close()
            ssh.close()
            print(f"  Pi 日志已下载: {pi_log}", flush=True)
        except Exception:
            print("  Pi 日志下载失败", flush=True)

        httpd.shutdown()
        print("Pi 追踪已停止，Web 已关闭", flush=True)


if __name__ == "__main__":
    main()
