#!/usr/bin/env python3
"""
三相机外参标定脚本
==================
通过 SSH 从 Pi 获取图像（usb1, usb2），本机直连 usb3，
用地面 AprilTag + solvePnP 计算每台相机的外参。

每次按 'c' 采集所有相机图像并标定，按 's' 保存，按 'q' 退出。
"""

import cv2, yaml, time, sys, os
import numpy as np
from pathlib import Path
from pupil_apriltags import Detector

PI_HOST = "192.168.3.17"
PI_USER = "pi"
PI_PASS = "alcht0"

# ==============================================================
# 工具
# ==============================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_floor_tags(path="floor_tags.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {int(k): (v["x"], v["y"], v["z"]) for k, v in raw["tags"].items()}

def cam_intrinsics(cfg):
    cm = cfg["camera_matrix"]
    K = np.array([[cm["fx"], 0, cm["cx"]],
                  [0, cm["fy"], cm["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(cfg["dist_coeffs"], dtype=np.float64)
    return K, dist

def calibrate_extrinsics(detections, floor_tag_map, tag_size, K, dist):
    """solvePnP 求解外参。"""
    obj_pts, img_pts = [], []
    half = tag_size / 2.0
    for d in detections:
        tid = d.tag_id
        if tid not in floor_tag_map:
            continue
        x, y, z = floor_tag_map[tid]
        obj_pts.append([x - half, y - half, z])
        obj_pts.append([x + half, y - half, z])
        obj_pts.append([x + half, y + half, z])
        obj_pts.append([x - half, y + half, z])
        img_pts.extend(d.corners)

    if len(obj_pts) < 4:
        return None, 0

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None, 0
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # 重投影误差
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1))
    return (R, t), err

# ==============================================================
# 捕获
# ==============================================================

def capture_pi_cameras(camera_names, config):
    """SSH 到 Pi 捕获指定相机图像。返回 dict[name -> np.ndarray]"""
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

    # 生成捕获脚本
    lines = ["import cv2, time"]
    for name in camera_names:
        cfg = next(c for c in config["cameras"] if c["name"] == name)
        w, h = cfg.get("resolution", [2560, 1440])
        dev = cfg["device"]
        lines.extend([
            f"cap = cv2.VideoCapture({dev}, cv2.CAP_V4L2)",
            f"cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))",
            f"cap.set(cv2.CAP_PROP_FRAME_WIDTH, {w})",
            f"cap.set(cv2.CAP_PROP_FRAME_HEIGHT, {h})",
            f"time.sleep(0.5)",
            f"[cap.read() for _ in range(10)]",
            f"ret, frame = cap.read()",
            f"if ret: cv2.imwrite('/tmp/calib_{name}.jpg', frame)",
            f"cap.release()",
            f"if ret: print('{name}: OK ' + str(frame.shape))",
            f"else: print('{name}: FAILED')",
        ])

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/calib_cap.py", "w") as f:
        f.write("\n".join(lines))
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command("python3 /tmp/calib_cap.py", timeout=30)
    print(stdout.read().decode().strip())
    ssh.close()

    # 下载
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    sftp = ssh.open_sftp()
    images = {}
    for name in camera_names:
        local_path = f"calib_{name}.jpg"
        try:
            sftp.get(f"/tmp/calib_{name}.jpg", local_path)
            img = cv2.imread(local_path)
            if img is not None:
                images[name] = img
        except Exception as e:
            print(f"  {name}: {e}")
    sftp.close()
    ssh.close()
    return images

def capture_pc_camera(name, config):
    """本机捕获。返回 (name, img) 或 (name, None)"""
    cfg = next(c for c in config["cameras"] if c["name"] == name)
    w, h = cfg.get("resolution", [2560, 1440])
    dev = int(cfg["device"])

    cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(0.5)
    for _ in range(10): cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret:
        print(f"  {name}: OK {frame.shape[1]}x{frame.shape[0]}")
        return frame
    else:
        print(f"  {name}: FAILED")
        return None

# ==============================================================
# 主流程
# ==============================================================

def main():
    config = load_config()
    floor_tags = load_floor_tags()
    tag_size = config["floor_tag_size"]
    detector = Detector(families="tag36h11", quad_decimate=1.0)
    clahe = cv2.createCLAHE(2.0, (8, 8))

    # 分组
    all_cams = [c["name"] for c in config["cameras"]]
    pi_cams = [c["name"] for c in config["cameras"] if c.get("host") == "pi"]
    pc_cams = [c["name"] for c in config["cameras"] if c.get("host") == "pc"]

    print(f"相机: Pi={pi_cams}  PC={pc_cams}")
    print(f"地面 Tag: {len(floor_tags)} 个")
    print("\n操作: 按 'c' 采集+标定 | 按 's' 保存 | 按 'q' 退出\n")

    extrinsics = {}  # name -> (R, t)
    last_images = {}  # name -> annotated image for display

    while True:
        key = input("> ").strip().lower()

        if key == 'q':
            break

        if key == 'c':
            print("\n采集所有相机...")
            images = {}

            # Pi 相机
            if pi_cams:
                print("  SSH Pi...")
                pi_images = capture_pi_cameras(pi_cams, config)
                images.update(pi_images)

            # PC 相机
            for name in pc_cams:
                print(f"  本机 {name}...")
                img = capture_pc_camera(name, config)
                if img is not None:
                    images[name] = img

            if not images:
                print("  没有获取到任何图像！")
                continue

            # 逐台标定
            print("\n标定结果:")
            for name in all_cams:
                if name not in images:
                    print(f"  [{name}] 无图像，跳过")
                    continue

                cfg = next(c for c in config["cameras"] if c["name"] == name)
                K, dist = cam_intrinsics(cfg)
                frame = images[name]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 降采样加速检测
                scale = 0.5 if max(frame.shape) > 2000 else 1.0
                gray_s = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
                gray_s = clahe.apply(gray_s)

                dets = detector.detect(gray_s)
                if scale != 1.0:
                    for d in dets:
                        d.corners /= scale
                        d.center = (d.center[0] / scale, d.center[1] / scale)

                floor_dets = [d for d in dets if d.tag_id in floor_tags]
                floor_ids = [d.tag_id for d in floor_dets]

                (R, t), err = calibrate_extrinsics(floor_dets, floor_tags, tag_size, K, dist)

                # 标注图像
                annotated = frame.copy()
                for d in dets:
                    pts = d.corners.astype(int)
                    is_floor = d.tag_id in floor_tags
                    color = (0, 255, 0) if is_floor else (0, 120, 120)
                    cv2.polylines(annotated, [pts], True, color, 2)
                    cx, cy = pts.mean(axis=0).astype(int)
                    cv2.putText(annotated, f"ID:{d.tag_id}", (cx, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if R is not None and err < 15:
                    extrinsics[name] = (R, t)
                    pos = (-R.T @ t).flatten()
                    cv2.putText(annotated, f"OK err={err:.2f}px", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(annotated, f"Pos:({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    print(f"  [{name}] OK  err={err:.2f}px  "
                          f"Tag:{floor_ids}  "
                          f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
                else:
                    cv2.putText(annotated, f"FAIL err={err:.2f}px tags={len(floor_dets)}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print(f"  [{name}] FAIL  err={err:.2f}px  "
                          f"地面Tag:{len(floor_dets)}")

                last_images[name] = annotated

            # 显示标注后的图像
            for name in all_cams:
                if name in last_images:
                    img_small = cv2.resize(last_images[name], (640, 360))
                    cv2.imshow(name, img_small)

            print(f"\n已标定: {len(extrinsics)}/{len(all_cams)} 台相机\n")

        if key == 's':
            if not extrinsics:
                print("还没有标定结果，先按 'c' 采集")
                continue

            out = {}
            for name in all_cams:
                if name in extrinsics:
                    R, t = extrinsics[name]
                    out[name] = {"R": R.tolist(), "t": t.flatten().tolist()}

            with open("extrinsics.yaml", "w") as f:
                yaml.dump(out, f, default_flow_style=None, allow_unicode=True)

            print(f"已保存 extrinsics.yaml ({len(extrinsics)} 台相机)")
            # 清屏标定结果
            print(f"相机位置:")
            for name in all_cams:
                if name in extrinsics:
                    R, t = extrinsics[name]
                    pos = (-R.T @ t).flatten()
                    print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    cv2.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    main()
