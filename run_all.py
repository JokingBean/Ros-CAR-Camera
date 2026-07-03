#!/usr/bin/env python3
"""
一键测试脚本 — 三相机 BEV + 追踪
================================
适配新 3 台 USB 相机 (Pi: video0/video2, PC: idx=1)
流式：抓图 → 标定 → BEV → 报告
"""

import os, sys, time, cv2, yaml, json
from datetime import datetime
from pathlib import Path
import numpy as np

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def pi_capture(cameras):
    """SSH 到 Pi 抓取指定相机图像。cameras: [(name, device_idx), ...]"""
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

    # 生成捕获脚本（全部自动曝光）
    lines = ["import cv2, time"]
    for name, idx in cameras:
        lines.append(f"cap = cv2.VideoCapture({idx}, cv2.CAP_V4L2)")
        lines.append("cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))")
        lines.append("cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)")
        lines.append("cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)")
        lines.append("# 一致曝光参数")
        lines.append("cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)")
        lines.append("cap.set(cv2.CAP_PROP_CONTRAST, 32)")
        lines.append("cap.set(cv2.CAP_PROP_GAMMA, 100)")
        lines.append("time.sleep(0.5)")
        lines.append("# skip dark frames, auto-exposure settles")
        lines.append("for _ in range(20):")
        lines.append("    ret, frame = cap.read()")
        lines.append("    if ret and frame.mean() > 10: break")
        lines.append("if ret and frame.mean() > 10:")
        lines.append("    cv2.imwrite('/tmp/" + name + ".jpg', frame)")
        lines.append("    print('" + name + ": OK ' + str(frame.shape[1]) + 'x' + str(frame.shape[0]) + ' mean=' + str(int(frame.mean())))")
        lines.append("else:")
        lines.append("    print('" + name + ": FAILED')")
        lines.append("cap.release()")
    lines.append("print('DONE')")

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/pi_all.py", "w") as f:
        f.write("\n".join(lines))
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command("python3 /tmp/pi_all.py", timeout=30)
    out = stdout.read().decode()
    ssh.close()

    success = "DONE" in out
    for line in out.splitlines():
        if "OK" in line or "FAILED" in line:
            print(f"  {line.strip()}")

    # 下载
    if success:
        time.sleep(1)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        sftp = ssh.open_sftp()
        images = {}
        for name, _ in cameras:
            try:
                sftp.get(f"/tmp/{name}.jpg", f"{TS}_{name}.jpg")
                img = cv2.imread(f"{TS}_{name}.jpg")
                if img is not None:
                    images[name] = img
            except Exception as e:
                print(f"  {name}: 下载失败 - {e}")
        sftp.close()
        ssh.close()
        return images
    return {}


def pc_capture(name, idx):
    """本机 USB 相机，降亮度匹配 Pi。"""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -20)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_GAMMA, 200)
    time.sleep(0.5)
    for _ in range(20):
        ret, frame = cap.read()
        if ret and frame.mean() > 10:
            break
    cap.release()
    if ret and frame.mean() > 10:
        cv2.imwrite(f"{TS}_{name}.jpg", frame)
        print(f"  {name}: OK {frame.shape[1]}x{frame.shape[0]} mean={frame.mean():.0f}")
        return frame
    print(f"  {name}: FAILED")
    return None


# ================================================================
def main():
    # ============================================================
    step("1/4  抓取三台相机图像")
    # ============================================================
    print("  Pi (usb1 + usb2 + usb3)...")
    pi_imgs = pi_capture([("usb1", 0), ("usb2", 2), ("usb3", 4)])
    images = {**pi_imgs}

    if len(images) == 0:
        print("[!] 没有捕获到任何图像")
        sys.exit(1)
    print(f"  共捕获 {len(images)} 台相机")
    for name, img in images.items():
        print(f"    {name}: {img.shape[1]}x{img.shape[0]} mean={img.mean():.1f}")

    # ============================================================
    step("2/4  自动外参标定")
    # ============================================================
    from pupil_apriltags import Detector
    with open("floor_tags.yaml", "r", encoding="utf-8") as f:
        ft = yaml.safe_load(f)
    floor_tags = {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
                  for k, v in ft["tags"].items()}
    CART_TAGS = {0, 1, 2, 3}
    HALF_TAG = 0.045

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def auto_calib(name, img):
        if img is None:
            return f"{name}: 无图像"
        # 获取该相机内参
        cc = next(c for c in cfg["cameras"] if c["name"] == name)
        cm = cc["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(cc["dist_coeffs"], dtype=np.float64)

        ih, iw = img.shape[:2]
        scale = 0.5 if iw > 2000 else 1.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray)
        if scale != 1.0:
            for d in dets:
                d.corners /= scale
                d.center = (d.center[0] / scale, d.center[1] / scale)

        good = [d for d in dets
                if d.tag_id in floor_tags and d.tag_id not in CART_TAGS
                and 0.05 * iw < d.center[0] < 0.95 * iw
                and 0.05 * ih < d.center[1] < 0.95 * ih]
        if len(good) < 6:
            return f"{name}: Tag不足 ({len(good)})"

        obj_pts, img_pts = [], []
        for d in good:
            wpt = floor_tags[d.tag_id]
            c3 = np.array([
                [wpt[0] - HALF_TAG, wpt[1] - HALF_TAG, 0],
                [wpt[0] + HALF_TAG, wpt[1] - HALF_TAG, 0],
                [wpt[0] + HALF_TAG, wpt[1] + HALF_TAG, 0],
                [wpt[0] - HALF_TAG, wpt[1] + HALF_TAG, 0],
            ], dtype=np.float64)
            for ci, ii in zip(c3, d.corners):
                obj_pts.append(ci)
                img_pts.append(ii)

        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)

        ok, rv, tv, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist,
            reprojectionError=4.0, confidence=0.99, iterationsCount=2000)
        if not ok:
            return f"{name}: PnP失败"

        R, _ = cv2.Rodrigues(rv)
        t = tv.flatten()
        pos = -R.T @ tv
        n_in = len(inl) if inl is not None else 0

        proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, dist)
        errs = [np.linalg.norm(proj[i].flatten() - img_pts[i]) for i in range(len(obj_pts))]

        return {
            "name": name,
            "R": R.tolist(),
            "t": t.tolist(),
            "tags": len(good),
            "err": round(float(np.mean(errs)), 1),
            "inliers": n_in,
            "height_cm": round(abs(float(pos[2]) * 100)),
        }

    import yaml as _y
    with open("extrinsics.yaml", "r") as f:
        ext = _y.safe_load(f) or {}
    calib_results = []
    for name in images:
        r = auto_calib(name, images[name])
        if isinstance(r, dict):
            ext[name] = {"R": r["R"], "t": r["t"]}
            calib_results.append(r)
            print(f"  {name}: {r['tags']}tags err={r['err']:.1f}px "
                  f"inliers={r['inliers']}/{r['tags']*4} H={r['height_cm']:.0f}cm")
        else:
            print(f"  {r}")

    with open("extrinsics.yaml", "w") as f:
        _y.dump(ext, f, default_flow_style=None)
    print(f"  外参已保存: {len(ext)} 台")

    # ============================================================
    step("3/4  生成 BEV 俯视图 (含单独相机)")
    # ============================================================
    from bev_generic import BevGenerator
    from pupil_apriltags import Detector
    gen = BevGenerator()
    active = list(images.keys())

    # 生成融合 BEV (同时返回各自相机的 BEV)
    fused, tag_data, cam_stats, per_cam_bevs, _masks = gen.run(camera_names=active, images=images)

    lit = (fused.sum(axis=2) > 0).sum()
    total = fused.shape[0] * fused.shape[1]
    print(f"  融合 BEV 覆盖: {100 * lit / total:.1f}%")
    print(f"  地面 Tag: {len(tag_data)} 个")
    n3 = sum(1 for t in tag_data if t["n_visible"] == len(active))
    n2 = sum(1 for t in tag_data if t["n_visible"] >= 2)
    print(f"  {len(active)}cam 重叠: {n3}  2cam+重叠: {n2}")

    for n, s in cam_stats.items():
        print(f"    {gen._name_to_label(n)}: {s['coverage_pct']}%")

    # ============================================================
    step("4/4  生成报告")
    # ============================================================
    cv2.imwrite(f"bev_{TS}_fused.jpg", fused)

    # 保存原图 + 单独相机 BEV
    for name in active:
        cv2.imwrite(f"bev_{TS}_{name}_raw.jpg", images[name])
        cv2.resize(images[name], (640, 360))  # 缩略图用于报告
        cv2.imwrite(f"bev_{TS}_{name}_thumb.jpg", cv2.resize(images[name], (640, 360)))
        if per_cam_bevs.get(name) is not None:
            cv2.imwrite(f"bev_{TS}_{name}_bev.jpg", per_cam_bevs[name])

    # HTML
    from collections import Counter
    best_counter = Counter(t["best_camera"] for t in tag_data)

    params = gen.get_camera_params(active)
    rows = ""
    for t in tag_data:
        gsd_cells = "".join(f"<td>{t['gsd_by_cam'].get(n, '-')}</td>" for n in active)
        best_n = t["best_camera"]
        best_l = gen._name_to_label(best_n)
        c = params[best_n]["color"]
        hex_ = f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}"
        rows += f'<tr><td>{t["id"]}</td><td>{t["x"]:.2f}</td><td>{t["y"]:.2f}</td><td>{t["n_visible"]}</td>{gsd_cells}<td style="color:{hex_};font-weight:bold">{best_l}</td></tr>\n'

    gsd_headers = "".join(f"<th>{gen._name_to_label(n)} GSD</th>" for n in active)

    cards = ""
    for n in active:
        c = params[n]["color"]
        hex_ = f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}"
        cards += f'<div class="card"><h3>{gen._name_to_label(n)} Coverage</h3><div class="v" style="color:{hex_}">{cam_stats[n]["coverage_pct"]}%</div></div>\n'
        cards += f'<div class="card"><h3>{gen._name_to_label(n)} Best GSD</h3><div class="v" style="color:{hex_}">{best_counter.get(n, 0)}</div></div>\n'

    legend = "".join(
        f'<span style="color:rgb({params[n]["color"][2]},{params[n]["color"][1]},{params[n]["color"][0]})">● {gen._name_to_label(n)}</span> '
        for n in active)

    per_cam_html = ""
    for name in active:
        c = params[name]["color"]
        # BEV
        if per_cam_bevs.get(name) is not None:
            cov = (per_cam_bevs[name].sum(axis=2) > 0).sum() / total * 100
            per_cam_html += f'<div style="flex:1;min-width:300px">\n'
            per_cam_html += f'<p style="color:rgb({c[2]},{c[1]},{c[0]});font-weight:bold;font-size:14px">● {gen._name_to_label(name)}</p>\n'
            # 原图缩略图
            per_cam_html += f'<p style="font-size:11px;color:#888;margin:2px 0">原图 ({images[name].shape[1]}x{images[name].shape[0]})</p>\n'
            per_cam_html += f'<img src="bev_{TS}_{name}_thumb.jpg" style="width:100%;border:1px solid #333" onclick="window.open(\'bev_{TS}_{name}_raw.jpg\')" title="点击看原图">\n'
            # BEV 图
            per_cam_html += f'<p style="font-size:11px;color:#888;margin:2px 0">BEV ({cov:.0f}% coverage)</p>\n'
            per_cam_html += f'<img src="bev_{TS}_{name}_bev.jpg" style="width:100%;border:1px solid #333">\n'
            per_cam_html += f'</div>\n'

    html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>3-Camera BEV Report</title>
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
</style></head><body>
<h1>3-Camera BEV Analysis</h1>
<p><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
<strong>Cameras:</strong> {', '.join(gen._name_to_label(n) for n in active)}</p>

<div class="cards">
<div class="card"><h3>Floor Tags</h3><div class="v">{len(tag_data)}</div></div>
<div class="card"><h3>{len(active)}-Cam Overlap</h3><div class="v">{n3}</div></div>
<div class="card"><h3>2-Cam+ Overlap</h3><div class="v">{n2}</div></div>
{cards}
</div>

<h2>融合 BEV ({100 * lit / total:.0f}% coverage)</h2>
<img src="bev_{TS}_fused.jpg" alt="Fused BEV">
<p>Tag dots: {legend} Larger dot = more cameras see it.</p>

<h2>各自相机 BEV</h2>
<div style="display:flex;gap:10px;flex-wrap:wrap;">
{per_cam_html}
</div>

<h2>Tag GSD (mm/px) — smaller = better</h2>
<table>
<tr><th>ID</th><th>X</th><th>Y</th><th>#Cam</th>{gsd_headers}<th>Best</th></tr>
{rows}
</table>
<div class="foot">Generated by run_all.py — homography-based BEV</div>
</body></html>'''

    with open(f"bev_{TS}_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  报告: bev_{TS}_report.html")
    print(f"  融合 BEV: bev_{TS}_fused.jpg")
    for name in active:
        print(f"  {name}: 原图 bev_{TS}_{name}_raw.jpg  |  BEV bev_{TS}_{name}_bev.jpg")

    import webbrowser
    webbrowser.open(f"bev_{TS}_report.html")


if __name__ == "__main__":
    main()
