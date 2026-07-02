#!/usr/bin/env python3
"""
精度测试脚本 — 立方体定位精度
=============================
把立方体放在地面网格点上，按 Enter 测量，自动对比真实位置计算误差。

用法:
  python precision_test.py                    # 交互式输入真实位置
  python precision_test.py --gt 2.0,2.5       # 指定真实位置
  python precision_test.py --auto             # 自动用地面 Tag 推算最近 0.5m 网格点
"""

import cv2, yaml, numpy as np, time, os, sys, json, argparse, paramiko
from datetime import datetime
from pupil_apriltags import Detector

PI_HOST = "192.168.3.17"
PI_USER = "pi"
PI_PASS = "alcht0"

# 立方体 Tag (ID 0-3)，边长 0.135m
TARGET_IDS = {0, 1, 2, 3}
TAG_SIZE = 0.135

# 网格参数
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = 0.0, 5.0
GRID_STEP = 0.5


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def capture_pi(cameras):
    """SSH 到 Pi 抓取指定相机。"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

    lines = ["import cv2, time"]
    for name, idx in cameras:
        lines.extend([
            f"cap = cv2.VideoCapture({idx}, cv2.CAP_V4L2)",
            "cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))",
            "cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)",
            "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)",
            "time.sleep(0.5)",
            "[cap.read() for _ in range(10)]",
            "ret, frame = cap.read()",
            f"if ret: cv2.imwrite('/tmp/pt_{name}.jpg', frame); print('{name}: OK')",
            f"else: print('{name}: FAILED')",
            "cap.release()",
        ])
    lines.append("print('DONE')")

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/pt_cap.py", "w") as f:
        f.write("\n".join(lines))
    sftp.close()
    stdin, stdout, stderr = ssh.exec_command("python3 /tmp/pt_cap.py", timeout=30)
    out = stdout.read().decode()
    ssh.close()

    success = "DONE" in out
    images = {}
    if success:
        time.sleep(1)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        sftp = ssh.open_sftp()
        for name, _ in cameras:
            try:
                sftp.get(f"/tmp/pt_{name}.jpg", f"_pt_{name}.jpg")
                img = cv2.imread(f"_pt_{name}.jpg")
                if img is not None:
                    images[name] = img
            except:
                pass
        sftp.close()
        ssh.close()
    return images


def capture_pc(name, idx):
    """本机抓取。"""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -20)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_GAMMA, 200)
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret and frame.mean() > 10:
        return frame
    return None


def detect_cube_homography(img, homography):
    """用 homography 将立方体 Tag 图像位置映射到地面 XY，不依赖外参。

    homography: 3x3 矩阵, world_xy -> image_uv, 从地面 Tag 计算。
    返回 [(tag_id, center_xy, gsd), ...]
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

    H_inv = np.linalg.inv(homography)
    results = []

    for d in dets:
        if d.tag_id not in TARGET_IDS:
            continue
        # homography 直接映射：图像像素 -> 世界 XY
        u, v = d.center
        wh = H_inv @ np.array([u, v, 1.0])
        wx, wy = wh[0] / wh[2], wh[1] / wh[2]

        results.append({
            "tag_id": d.tag_id,
            "center_xy": [float(wx), float(wy)],
            "gsd": 0.0,  # homography 模式不提供 GSD，用 Tag 像素尺寸替代
            "diag_px": float(np.linalg.norm(d.corners[0] - d.corners[2])),
        })

    return results


def grid_snap(x, y, step=GRID_STEP):
    """吸附到最近网格点。"""
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))

def detect_cube_extrinsics(img, K, dist, R, t):
    """回退方案：用外参 solvePnP 定位立方体 Tag。"""
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
        Rt, _ = cv2.Rodrigues(rvec)
        tt = tvec.reshape(3, 1)
        Rc = R.T
        tc = -Rc @ t
        tw = (Rc @ tt + tc).flatten()
        Z_tag = Rt[:, 2]
        Z_world = Rc @ Z_tag
        Z_world = Z_world / np.linalg.norm(Z_world)
        center = tw - 0.125 * Z_world
        P = R @ tw.reshape(3, 1) + t
        gsd = np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000
        results.append({
            "tag_id": d.tag_id,
            "center_xy": [float(center[0]), float(center[1])],
            "gsd": round(float(gsd), 2),
        })
    return results
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))


def main():
    parser = argparse.ArgumentParser(description="立方体定位精度测试")
    parser.add_argument("--auto", action="store_true", default=True,
                        help="自动吸附到最近 0.5m 网格")
    args = parser.parse_args()

    # 建立总文件夹
    runs_dir = "precision_runs"
    os.makedirs(runs_dir, exist_ok=True)

    config = load_config()
    all_cams = [c["name"] for c in config["cameras"]]
    cam_params = {}
    with open("extrinsics.yaml", "r") as f:
        ext = yaml.safe_load(f)

    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        R = np.array(ext[name]["R"])
        t = np.array(ext[name]["t"]).reshape(3, 1)
        cam_params[name] = (K, dist, R, t)

    # 汇总文件
    summary_file = os.path.join(runs_dir, "_summary.jsonl")
    all_measurements = []

    # 加载已有记录
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            for line in f:
                try: all_measurements.append(json.loads(line.strip()))
                except: pass

    print("=" * 50)
    print("  精度测试 — 立方体定位")
    print(f"  网格: {X_MIN}-{X_MAX}m x {Y_MIN}-{Y_MAX}m, 步长 {GRID_STEP}m")
    print(f"  结果目录: {runs_dir}/")
    print("  按 Enter 测量 | 'q' 退出 | 's' 看统计")
    print("=" * 50)

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == 'q':
            break
        if cmd == 's':
            if all_measurements:
                errs = [m["error_xy_cm"] for m in all_measurements]
                print(f"\n  已测 {len(all_measurements)} 次")
                print(f"  平均误差: {np.mean(errs):.1f}cm")
                print(f"  最大误差: {np.max(errs):.1f}cm")
                print(f"  最小误差: {np.min(errs):.1f}cm")
            else:
                print("  还没有测量数据")
            continue

        # 创建本次测量文件夹
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_dir, ts)
        os.makedirs(run_dir, exist_ok=True)

        # 抓图
        print("  抓图中...")
        images = capture_pi([("usb1", 0), ("usb2", 2)])
        pc_img = capture_pc("usb3", 1)
        if pc_img is not None:
            images["usb3"] = pc_img

        # 保存原图
        for name, img in images.items():
            cv2.imwrite(os.path.join(run_dir, f"{name}.jpg"), img)

        if not images:
            print("  未捕获到任何图像")
            continue

        # 检测立方体 — 用 homography 定位（不依赖外参）
        all_results = []
        log_lines = [f"=== 精度测试 {ts} ===", ""]

        # 先算每台相机的 homography（从地面 Tag）
        with open("floor_tags.yaml", "r", encoding="utf-8") as f:
            ft = yaml.safe_load(f)
        floor_tags = {int(k): (v["x"], v["y"]) for k, v in ft["tags"].items()}

        homographies = {}
        detector = Detector(families="tag36h11", quad_decimate=1.0)
        clahe = cv2.createCLAHE(2.0, (8, 8))

        for name in all_cams:
            if name not in images:
                continue
            img = images[name]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = 0.5 if max(img.shape) > 2000 else 1.0
            gray_s = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
            gray_s = clahe.apply(gray_s)
            fdets = detector.detect(gray_s)
            for d in fdets:
                d.corners /= scale
                d.center = (d.center[0] / scale, d.center[1] / scale)
            fd = [d for d in fdets if d.tag_id in floor_tags]
            if len(fd) >= 4:
                wxy = np.array([floor_tags[d.tag_id] for d in fd], dtype=np.float64)
                iuv = np.array([d.center for d in fd], dtype=np.float64)
                H, _ = cv2.findHomography(wxy, iuv, cv2.RANSAC, 5.0)
                if H is not None:
                    homographies[name] = H

        for name in all_cams:
            if name not in images:
                continue
            if name in homographies:
                results = detect_cube_homography(images[name], homographies[name])
            else:
                # 回退到外参法
                K, dist, R, t = cam_params[name]
                results = detect_cube_extrinsics(images[name], K, dist, R, t)
            for r in results:
                all_results.append((name, r))
                cx, cy = r["center_xy"]
                w = r.get("gsd", 0) or r.get("diag_px", 0)
                line = (f"{name} T{r['tag_id']}: "
                        f"xy=({cx:.3f},{cy:.3f}) {'GSD='+str(w)+'mm' if r.get('gsd') else 'diag='+str(int(w))+'px'}")
                print(f"  {line}")
                log_lines.append(line)

        if not all_results:
            print("  未检测到立方体 Tag (ID 0-3)")
            log_lines.append("NO CUBE DETECTED")
            with open(os.path.join(run_dir, "log.txt"), "w") as f:
                f.write("\n".join(log_lines))
            continue

        # 融合 (按像素尺寸或 GSD 加权)
        weights = []
        for r in all_results:
            if r[1].get("gsd", 0) > 0:
                weights.append(1.0 / max(r[1]["gsd"], 0.01))
            else:
                weights.append(r[1].get("diag_px", 10) ** 2)  # 像素面积
        weights = np.array(weights)
        weights /= weights.sum()
        xys = np.array([r[1]["center_xy"] for r in all_results])
        fused_xy = np.average(xys, axis=0, weights=weights)

        print(f"\n  融合: ({fused_xy[0]:.3f}, {fused_xy[1]:.3f})  [{len(all_results)} obs]")
        log_lines.append(f"\nFUSED: ({fused_xy[0]:.3f}, {fused_xy[1]:.3f})  [{len(all_results)} obs]")

        # 自动吸附
        gx, gy = grid_snap(fused_xy[0], fused_xy[1])
        print(f"  吸附: ({gx:.1f}, {gy:.1f})m")
        log_lines.append(f"SNAP: ({gx:.1f}, {gy:.1f})m")

        # 误差
        err_xy = np.linalg.norm([fused_xy[0] - gx, fused_xy[1] - gy]) * 100
        print(f"  误差 XY: {err_xy:.1f}cm")
        log_lines.append(f"ERROR XY: {err_xy:.1f}cm")

        record = {
            "time": ts,
            "folder": run_dir,
            "ground_truth": [gx, gy],
            "fused_xy": [round(float(fused_xy[0]), 3),
                         round(float(fused_xy[1]), 3)],
            "error_xy_cm": round(float(err_xy), 1),
            "n_obs": len(all_results),
            "method": "homography",
            "per_camera": {},
        }
        for name, r in all_results:
            cx, cy = r["center_xy"]
            e_xy = np.linalg.norm([cx - gx, cy - gy]) * 100
            record["per_camera"][name] = {
                "tag_id": r["tag_id"],
                "xy": [round(float(cx), 3), round(float(cy), 3)],
                "error_xy_cm": round(float(e_xy), 1),
                "diag_px": round(r.get("diag_px", 0), 1),
            }
            log_lines.append(f"  {name} T{r['tag_id']}: xy=({cx:.3f},{cy:.3f}) error={e_xy:.1f}cm")

        # 保存本文件夹
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2)
        with open(os.path.join(run_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

        # 追加到汇总
        all_measurements.append(record)
        with open(summary_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  已保存 → {run_dir}/  ({len(all_measurements)} 条累计)")

    # 最终统计
    if all_measurements:
        errs = [m["error_xy_cm"] for m in all_measurements]
        print(f"\n{'='*50}")
        print(f"  统计 ({len(all_measurements)} 次)")
        print(f"  平均: {np.mean(errs):.1f}cm  |  最大: {np.max(errs):.1f}cm  |  最小: {np.min(errs):.1f}cm")
        print(f"  汇总: {summary_file}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
