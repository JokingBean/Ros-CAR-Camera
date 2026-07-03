#!/usr/bin/env python3
"""
连续精度测试 — 立方体实时定位
==============================
摄像头常开，连续抓图检测，实时显示位置和帧率。
Ctrl+C 停止。
"""

import cv2, yaml, numpy as np, time, os, sys, paramiko
from collections import deque
from datetime import datetime

from src.tracking import detect_cube_extrinsics, grid_snap, TARGET_IDS, TAG_SIZE

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"

# 分辨率
W, H = 2560, 1440


def open_pi_camera(idx):
    """通过 SSH 在 Pi 上打开一个摄像头，通过 MJPEG stream 或连续抓帧返回。
    这里用最简单的方案：轮询抓图。"""
    pass  # Pi 端持续抓帧走 socket/tcp 会更好，先保留接口


def main():
    config_file = "cfg/config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open("cfg/extrinsics.yaml", "r") as f:
        ext = yaml.safe_load(f)

    cam_params = {}
    for c in config["cameras"]:
        name = c["name"]
        cm = c["camera_matrix"]
        K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1]], dtype=np.float64)
        dist = np.array(c["dist_coeffs"], dtype=np.float64)
        R = np.array(ext[name]["R"]) if name in ext else np.eye(3)
        t = np.array(ext[name]["t"]).reshape(3, 1) if name in ext else np.array([[0], [0], [1.5]])
        cam_params[name] = (K, dist, R, t)

    # 分组：Pi 走 SSH，本机直连
    pi_cams = [(c["name"], int(c["device"])) for c in config["cameras"] if c.get("host") == "pi"]
    pc_cams = [(c["name"], int(c["device"])) for c in config["cameras"] if c.get("host") != "pi"]

    print("=" * 60)
    print("  连续精度测试 — 实时定位")
    if pi_cams:
        print(f"  Pi: {[n for n,_ in pi_cams]} (SSH)")
    if pc_cams:
        print(f"  PC: {[n for n,_ in pc_cams]} (直连)")
    print("  Ctrl+C 停止")
    print("=" * 60)

    # 本机相机打开（常驻）
    pc_caps = {}
    for name, idx in pc_cams:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -20)
        cap.set(cv2.CAP_PROP_CONTRAST, 40)
        cap.set(cv2.CAP_PROP_GAMMA, 200)
        time.sleep(0.5)
        for _ in range(10):
            cap.read()
        pc_caps[name] = cap
        print(f"  {name}: 已打开")

    # Pi 相机持续抓帧脚本
    ssh = None
    if pi_cams:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
        # 上传一次性抓帧脚本
        sftp = ssh.open_sftp()
        script = "import cv2, time, sys\n"
        for name, idx in pi_cams:
            script += f"""
cap_{idx} = cv2.VideoCapture({idx}, cv2.CAP_V4L2)
cap_{idx}.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_{idx}.set(cv2.CAP_PROP_FRAME_WIDTH, {W})
cap_{idx}.set(cv2.CAP_PROP_FRAME_HEIGHT, {H})
cap_{idx}.set(cv2.CAP_PROP_BRIGHTNESS, 30)
cap_{idx}.set(cv2.CAP_PROP_CONTRAST, 40)
cap_{idx}.set(cv2.CAP_PROP_GAMMA, 100)
time.sleep(0.3)
for _ in range(10): cap_{idx}.read()
"""
        script += "\nwhile True:\n  try:\n"
        for name, idx in pi_cams:
            script += f"    ret, frame = cap_{idx}.read()\n"
            script += f"    if ret: cv2.imwrite('/tmp/live_{name}.jpg', frame)\n"
        script += "  except: break\n"
        with sftp.file("/tmp/live_cap.py", "w") as f:
            f.write(script)
        sftp.close()
        # 后台启动
        ssh.exec_command("nohup python3 /tmp/live_cap.py >/dev/null 2>&1 &")
        print(f"  Pi: 持续抓帧中...")

    fps_history = deque(maxlen=30)
    pos_history = deque(maxlen=5)

    try:
        while True:
            t0 = time.time()
            frames = {}

            # 本机读帧
            for name, cap in pc_caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame

            # Pi 读帧（SFTP 下载最新文件）
            if pi_cams and ssh:
                try:
                    sftp2 = ssh.open_sftp()
                    for name, idx in pi_cams:
                        try:
                            sftp2.get(f"/tmp/live_{name}.jpg", f"_live_{name}.jpg")
                            img = cv2.imread(f"_live_{name}.jpg")
                            if img is not None:
                                frames[name] = img
                        except:
                            pass
                    sftp2.close()
                except:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
                    ssh.exec_command("nohup python3 /tmp/live_cap.py >/dev/null 2>&1 &")

            if not frames:
                time.sleep(0.01)
                continue

            # 检测 + 融合
            all_results = []
            for name, frame in frames.items():
                if name not in cam_params:
                    continue
                K, dist, R, t = cam_params[name]
                results = detect_cube_extrinsics(frame, K, dist, R, t)
                all_results.extend([(name, r) for r in results])

            t_elapsed = (time.time() - t0) * 1000
            fps_history.append(1000 / t_elapsed if t_elapsed > 0 else 0)
            avg_fps = np.mean(fps_history) if fps_history else 0

            if all_results:
                good = [r for r in all_results if r[1].get("margin", 0) >= 20] or all_results
                xys = np.array([r[1]["center_xy"] for r in good])
                gsds = np.array([r[1].get("gsd", 1.0) for r in good])
                w = 1.0 / np.maximum(gsds, 0.01)
                w /= w.sum()
                fused_xy = np.average(xys, axis=0, weights=w)
                pos_history.append(fused_xy)
                smooth_xy = np.mean(np.array(pos_history), axis=0)
                gx, gy = grid_snap(smooth_xy[0], smooth_xy[1])
                err = np.linalg.norm([smooth_xy[0] - gx, smooth_xy[1] - gy]) * 100

                tags_str = " ".join(f"{n}T{r['tag_id']}" for n, r in all_results)
                line = (f"\r  XY=({smooth_xy[0]:.3f},{smooth_xy[1]:.3f})  "
                        f"grid=({gx:.1f},{gy:.1f})  err={err:.1f}cm  "
                        f"FPS={avg_fps:.1f}  [{len(good)}/{len(all_results)} tags: {tags_str}]")
                print(line + " " * 10, end="", flush=True)
            else:
                print(f"\r  未检测到  FPS={avg_fps:.1f}  {len(frames)} cameras    ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  停止")
    finally:
        for cap in pc_caps.values():
            cap.release()
        if ssh:
            ssh.exec_command("pkill -f live_cap.py 2>/dev/null")
            ssh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
