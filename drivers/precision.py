#!/usr/bin/env python3
"""
精度测试 — 立方体定位
======================
把立方体放在 0.5m 网格点上，按 Enter 测量。
自动抓图 → 检测 → 融合 → 吸附网格 → 计算误差 → 保存。

图像存 Pi 本地, 精度结果存 PC 本机。
"""

import cv2, yaml, numpy as np, time, os, sys, json, paramiko
from datetime import datetime

from src.tracking import (detect_cube_extrinsics, grid_snap,
                           GRID_STEP, TARGET_IDS, TAG_SIZE)

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"
PI_DATA_DIR = "/home/pi/UwbCamera/data"
PC_DATA_DIR = "../precision_runs"

os.makedirs(PC_DATA_DIR, exist_ok=True)


def load_config():
    with open("cfg/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pi_capture(cameras, subdir):
    """SSH 到 Pi 抓图，保存到 Pi 本地文件夹。"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

    remote_dir = f"{PI_DATA_DIR}/{subdir}"
    script = f"""
import cv2, time, os
os.makedirs(\"{remote_dir}\", exist_ok=True)
"""
    for name, idx in cameras:
        script += f"""
cap = cv2.VideoCapture({idx}, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
cap.set(cv2.CAP_PROP_CONTRAST, 40)
cap.set(cv2.CAP_PROP_GAMMA, 100)
time.sleep(0.5)
for _ in range(10): cap.read()
ret, frame = cap.read()
if ret:
    cv2.imwrite(\"{remote_dir}/{name}.jpg\", frame)
    print(f\"{name}: OK {{frame.shape[1]}}x{{frame.shape[0]}} mean={{int(frame.mean())}}\")
else:
    print(f\"{name}: FAILED\")
cap.release()
"""
    script += "print('DONE')\n"

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/picap.py", "w") as f:
        f.write(script)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command("python3 /tmp/picap.py", timeout=30)
    out = stdout.read().decode()
    ssh.close()

    if "DONE" not in out:
        print("  Pi capture failed")
        return {}

    # 下载
    time.sleep(1)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    sftp = ssh.open_sftp()
    images = {}
    for name, _ in cameras:
        try:
            sftp.get(f"{remote_dir}/{name}.jpg", f"{PC_DATA_DIR}/{subdir}/{name}.jpg")
            img = cv2.imread(f"{PC_DATA_DIR}/{subdir}/{name}.jpg")
            if img is not None:
                images[name] = img
        except Exception as e:
            print(f"  {name}: download failed - {e}")
    sftp.close()
    ssh.close()
    return images


def main():
    config = load_config()
    all_cams = [c["name"] for c in config["cameras"]]
    cam_params = {}
    with open("cfg/extrinsics.yaml", "r") as f:
        ext = yaml.safe_load(f)
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        R = np.array(ext[name]["R"])
        t = np.array(ext[name]["t"]).reshape(3, 1)
        cam_params[name] = (K, dist, R, t)

    summary_file = os.path.join(PC_DATA_DIR, "_summary.jsonl")
    history = []
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            for line in f:
                try:
                    history.append(json.loads(line.strip()))
                except:
                    pass

    print("=" * 60)
    print("  精度测试 — 立方体定位")
    print(f"  网格间距: {GRID_STEP}m | 范围: 0-4.5m × 0-5m")
    print(f"  数据: {PC_DATA_DIR}/")
    print(f"  Enter 测量 | q 退出 | s 看统计")
    print("=" * 60)

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            if history:
                errs = [m["error_cm"] for m in history]
                print(f"\n  累计: {len(history)} 次")
                print(f"  误差: 平均 {np.mean(errs):.1f}cm  "
                      f"最大 {np.max(errs):.1f}cm  "
                      f"最小 {np.min(errs):.1f}cm")
            else:
                print("  还没有数据")
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(PC_DATA_DIR, ts)
        os.makedirs(run_dir, exist_ok=True)

        # ======== 抓图 + 计时 ========
        t0 = time.time()
        print("  [抓图] ", end="", flush=True)
        images = pi_capture([("usb1", 0), ("usb2", 2), ("usb3", 4)], ts)
        t_capture = (time.time() - t0) * 1000

        if not images:
            print("  未捕获到图像")
            continue

        n_captured = len(images)
        print(f"  ✓ {n_captured}/3 台  ({t_capture:.0f}ms)")

        # ======== 检测 ========
        t1 = time.time()
        all_results = []
        log_lines = [f"### 精度测试 {ts}", ""]
        log_lines.append(f"# 抓图: {t_capture:.0f}ms, {n_captured}/3 相机")
        log_lines.append("")

        for name in all_cams:
            if name not in images:
                continue
            K, dist, R, t = cam_params[name]
            results = detect_cube_extrinsics(images[name], K, dist, R, t)
            for r in results:
                all_results.append((name, r))
                tx, ty, tz = r["tag_3d"]
                cx, cy = r["center_xy"]
                diag = r["diag_px"]
                margin = r["margin"]
                line = (f"[{name}] Tag{r['tag_id']}  "
                        f"tag3D=({tx:.3f},{ty:.3f},{tz:.3f})  "
                        f"XY=({cx:.3f},{cy:.3f})  "
                        f"diag={diag:.0f}px  margin={margin:.1f}  GSD={r['gsd']}mm")
                print(f"  {line}")
                log_lines.append(line)
            if not results:
                log_lines.append(f"[{name}] {len(results)} tags")

        t_detect = (time.time() - t1) * 1000

        if not all_results:
            print(f"  未检测到立方体 Tag (ID {TARGET_IDS})")
            log_lines.append("NO CUBE DETECTED")
            with open(os.path.join(run_dir, "log.txt"), "w") as f:
                f.write("\n".join(log_lines))
            continue

        # ======== 融合 + 误差 ========
        good = [r for r in all_results if r[1].get("margin", 0) >= 20]
        if not good:
            good = all_results

        xys = np.array([r[1]["center_xy"] for r in good])
        gsds = np.array([r[1].get("gsd", 1.0) for r in good])
        weights = 1.0 / np.maximum(gsds, 0.01)
        weights /= weights.sum()

        median_xy = np.median(xys, axis=0)
        weighted_xy = np.average(xys, axis=0, weights=weights)
        fused_xy = median_xy

        gx, gy = grid_snap(fused_xy[0], fused_xy[1])
        err_xy = np.linalg.norm([fused_xy[0] - gx, fused_xy[1] - gy]) * 100
        t_total = (time.time() - t0) * 1000
        fps = 1000 / t_total if t_total > 0 else 0

        print(f"\n  [{len(good)}/{len(all_results)} 有效观测]")
        print(f"  中位数: ({fused_xy[0]:.3f}, {fused_xy[1]:.3f})")
        print(f"  加权:   ({weighted_xy[0]:.3f}, {weighted_xy[1]:.3f})")
        print(f"  网格: ({gx:.1f}, {gy:.1f})m  →  误差: {err_xy:.1f}cm")
        print(f"  ⏱ FPS: {fps:.1f}  (总 {t_total:.0f}ms = 抓图{t_capture:.0f} + 检测{t_detect:.0f} + 其他{t_total-t_capture-t_detect:.0f})")

        log_lines.append("")
        log_lines.append(f"## 融合 ({len(good)}/{len(all_results)} 有效)")
        log_lines.append(f"  中位数: ({fused_xy[0]:.3f}, {fused_xy[1]:.3f})")
        log_lines.append(f"  加权值: ({weighted_xy[0]:.3f}, {weighted_xy[1]:.3f})")
        log_lines.append(f"  网格点: ({gx:.1f}, {gy:.1f})m")
        log_lines.append(f"  误差: {err_xy:.1f}cm")
        log_lines.append(f"  FPS: {fps:.1f}  总耗时: {t_total:.0f}ms")
        log_lines.append("")
        log_lines.append("# 各相机融合权重:")
        for i, (name, r) in enumerate(good):
            log_lines.append(f"  [{name}] T{r['tag_id']} wt={weights[i]:.3f}  err={np.linalg.norm([r['center_xy'][0]-gx,r['center_xy'][1]-gy])*100:.1f}cm")

        # ======== 保存 ========
        record = {
            "time": ts,
            "grid": [gx, gy],
            "fused_xy": [round(float(fused_xy[0]), 3), round(float(fused_xy[1]), 3)],
            "error_cm": round(float(err_xy), 1),
            "obs_total": len(all_results),
            "obs_valid": len(good),
            "n_cams": n_captured,
            "fps": round(float(fps), 1),
            "capture_ms": round(t_capture),
            "detect_ms": round(t_detect),
            "total_ms": round(t_total),
            "tags": [],
        }
        for name, r in all_results:
            cx, cy = r["center_xy"]
            record["tags"].append({
                "camera": name,
                "tag_id": r["tag_id"],
                "tag_3d": r["tag_3d"],
                "center_xy": [round(cx, 3), round(cy, 3)],
                "gsd_mm": r["gsd"],
                "diag_px": round(r["diag_px"], 1),
                "margin": round(r["margin"], 1),
                "error_cm": round(np.linalg.norm([cx - gx, cy - gy]) * 100, 1),
                "weight": round(float(weights[all_results.index((name, r))]) if (name, r) in good else 0, 3),
            })

        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2)
        with open(os.path.join(run_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

        history.append(record)
        with open(summary_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  保存: {run_dir}/  ({len(history)} 条累计)")

    # ======== 统计 ========
    if history:
        errs = [m["error_cm"] for m in history]
        fpss = [m["fps"] for m in history]
        print(f"\n{'='*60}")
        print(f"  累计: {len(history)} 次")
        print(f"  误差:  avg {np.mean(errs):.1f}cm  max {np.max(errs):.1f}cm  min {np.min(errs):.1f}cm")
        print(f"  FPS:   avg {np.mean(fpss):.1f}  max {np.max(fpss):.1f}")
        print(f"  数据: {PC_DATA_DIR}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
