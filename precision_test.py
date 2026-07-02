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


def detect_cube(img, K, dist, R, t, tag_size):
    """检测图像中的立方体 Tag，返回 [(tag_id, world_center, gsd), ...]"""
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
    half = tag_size / 2.0
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

        # Tag 在相机坐标 → 世界坐标
        Rc = R.T
        tc = -Rc @ t
        tw = (Rc @ tt + tc).flatten()

        # 立方体中心 = Tag 位置 + 面偏移 (立方体边长 0.25m, 面到中心 = 0.125m)
        h_local = {0: np.array([1, 0, 0]), 1: np.array([0, 0, 1]),
                    2: np.array([-1, 0, 0]), 3: np.array([0, 0, -1])}.get(d.tag_id, np.array([0, 0, 1]))
        h_w = Rc @ Rt @ h_local
        h_w = h_w[:2] / np.linalg.norm(h_w[:2]) if np.linalg.norm(h_w[:2]) > 1e-6 else np.array([0, 1])
        side = np.array([h_w[1], -h_w[0]])
        sign = {0: -1, 1: -1, 2: 1, 3: 1}.get(d.tag_id, 0)
        offset = (h_w if d.tag_id in (1, 3) else side) * sign * 0.125
        center = tw + np.array([offset[0], offset[1], -0.125])

        # GSD
        P = R @ tw.reshape(3, 1) + t
        gsd = np.linalg.norm(P) / ((K[0, 0] + K[1, 1]) / 2) * 1000

        results.append({
            "tag_id": d.tag_id,
            "tag_pos": tw.tolist(),
            "center": center.tolist(),
            "gsd": round(float(gsd), 2),
        })

    return results


def grid_snap(x, y, step=GRID_STEP):
    """吸附到最近网格点。"""
    gx = round(x / step) * step
    gy = round(y / step) * step
    return max(X_MIN, min(X_MAX, gx)), max(Y_MIN, min(Y_MAX, gy))


def main():
    parser = argparse.ArgumentParser(description="立方体定位精度测试")
    parser.add_argument("--gt", type=str, help="真实位置，如 2.0,2.5")
    parser.add_argument("--auto", action="store_true", help="自动吸附到最近 0.5m 网格")
    parser.add_argument("--save", action="store_true", help="保存测量结果到文件")
    args = parser.parse_args()

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

    results_file = f"precision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    measurements = []

    print("精度测试 — 立方体定位")
    print(f"网格: {X_MIN}-{X_MAX}m x {Y_MIN}-{Y_MAX}m, 步长 {GRID_STEP}m")
    print("按 Enter 测量 | 输入 'q' 退出 | 输入 's' 保存\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == 'q':
            break
        if cmd == 's' and measurements:
            with open(results_file, 'w') as f:
                for m in measurements:
                    f.write(json.dumps(m) + "\n")
            print(f"已保存 {len(measurements)} 条到 {results_file}")
            continue

        # 确定真实位置
        if args.gt:
            parts = args.gt.split(",")
            gt_x, gt_y = float(parts[0]), float(parts[1])
        elif cmd and ',' in cmd:
            parts = cmd.split(",")
            gt_x, gt_y = float(parts[0]), float(parts[1])
        else:
            gt_x, gt_y = None, None

        # 抓图
        print("  抓图中...")
        images = capture_pi([("usb1", 0), ("usb2", 2)])
        pc_img = capture_pc("usb3", 1)
        if pc_img is not None:
            images["usb3"] = pc_img

        if not images:
            print("  未捕获到任何图像")
            continue

        # 检测立方体
        all_results = []
        print(f"\n  {'相机':<8} {'Tag':<4} {'Tag位置':<22} {'中心':<22} {'GSD':<8}")
        print(f"  {'-'*8} {'-'*4} {'-'*22} {'-'*22} {'-'*8}")

        for name in all_cams:
            if name not in images:
                continue
            K, dist, R, t = cam_params[name]
            results = detect_cube(images[name], K, dist, R, t, TAG_SIZE)
            for r in results:
                all_results.append((name, r))
                tp = r["tag_pos"]
                cp = r["center"]
                print(f"  {name:<8} T{r['tag_id']:<3d} "
                      f"({tp[0]:.3f},{tp[1]:.3f},{tp[2]:.3f})   "
                      f"({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})   "
                      f"{r['gsd']:.1f}mm")

        if not all_results:
            print("  未检测到立方体 Tag (ID 0-3)")
            continue

        # 融合
        weights = np.array([1.0 / max(r[1]["gsd"], 0.01) for r in all_results])
        weights /= weights.sum()
        centers = np.array([r[1]["center"] for r in all_results])
        fused_center = np.average(centers, axis=0, weights=weights)
        fused_gsd = min(r[1]["gsd"] for r in all_results)

        print(f"\n  融合中心: ({fused_center[0]:.3f}, {fused_center[1]:.3f}, {fused_center[2]:.3f})  "
              f"GSD={fused_gsd:.1f}mm  [{len(all_results)} 次观测]")

        # 自动吸附
        if args.auto or (gt_x is None and gt_y is None):
            gx, gy = grid_snap(fused_center[0], fused_center[1])
            print(f"  吸附网格: ({gx:.1f}, {gy:.1f})m")
            gt_x, gt_y = gx, gy

        # 误差
        if gt_x is not None and gt_y is not None:
            err_xy = np.linalg.norm([fused_center[0] - gt_x, fused_center[1] - gt_y]) * 100
            err_xyz = np.linalg.norm([fused_center[0] - gt_x, fused_center[1] - gt_y, fused_center[2]]) * 100
            print(f"\n  真实: ({gt_x:.2f}, {gt_y:.2f})m")
            print(f"  误差 XY: {err_xy:.1f}cm  |  XYZ: {err_xyz:.1f}cm")

            record = {
                "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "ground_truth": [gt_x, gt_y],
                "fused_center": [round(float(fused_center[0]), 3),
                                 round(float(fused_center[1]), 3),
                                 round(float(fused_center[2]), 3)],
                "error_xy_cm": round(float(err_xy), 1),
                "error_xyz_cm": round(float(err_xyz), 1),
                "n_observations": len(all_results),
                "per_camera": {},
            }
            for name, r in all_results:
                cp = np.array(r["center"])
                e_xy = np.linalg.norm(cp[:2] - [gt_x, gt_y]) * 100
                record["per_camera"][name] = {
                    "tag_id": r["tag_id"],
                    "center": [round(float(cp[0]), 3), round(float(cp[1]), 3), round(float(cp[2]), 3)],
                    "error_xy_cm": round(float(e_xy), 1),
                    "gsd_mm": r["gsd"],
                }
            measurements.append(record)
            print(f"  已记录 ({len(measurements)} 条)")
        print()

    # 统计
    if measurements:
        errs = [m["error_xy_cm"] for m in measurements]
        print(f"\n=== 统计 ({len(measurements)} 次) ===")
        print(f"  平均误差: {np.mean(errs):.1f}cm")
        print(f"  最大误差: {np.max(errs):.1f}cm")
        print(f"  最小误差: {np.min(errs):.1f}cm")

        with open(results_file, 'w') as f:
            for m in measurements:
                f.write(json.dumps(m) + "\n")
        print(f"  已保存: {results_file}")


if __name__ == "__main__":
    main()
