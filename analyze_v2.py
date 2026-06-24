"""
双相机快照分析 v2 — 使用真实配置
==================================
使用从树莓派拉取的 camera_intrinsics.yaml、cameras_config.yaml、
global_extrinsics.yaml、floor_tag_layout.yaml 分析 snapshots 中的配对图像。"""

import os, json, sys
import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

# ==============================================================
# 加载真实配置
# ==============================================================

def load_floor_tags(path="pi_config_floor_tag_layout.yaml"):
    """floor_tag_layout.yaml 实际是 JSON 格式。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # {tag_id: [x, y, z]}
    return {int(k): np.array(v, dtype=np.float64) for k, v in raw.items()}

def load_extrinsics(path="pi_config_global_extrinsics.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ext = {}
    for cam_name, data in raw["cameras"].items():
        R = np.array(data["R"], dtype=np.float64)
        t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        ext[cam_name] = (R, t)
    return ext

def load_camera_config(path="pi_config_cameras_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==============================================================
# 图片配对
# ==============================================================

def pair_snapshots(snap_dir="snapshots"):
    import re
    from collections import defaultdict
    pairs = defaultdict(dict)
    for f in sorted(os.listdir(snap_dir)):
        if not f.endswith(".jpg"):
            continue
        m = re.search(r"(\d{8}_\d{6})", f)
        if not m:
            continue
        ts = m.group(1)
        if "picam" in f:
            pairs[ts]["picam"] = os.path.join(snap_dir, f)
        elif "usb" in f:
            pairs[ts]["usb"] = os.path.join(snap_dir, f)
    return dict(pairs)

# ==============================================================
# 可视化工具
# ==============================================================

def draw_tags(img, detections, floor_ids, common_ids=None):
    """标注 Tag：共视=红，地面=绿，其他=蓝"""
    ann = img.copy()
    common_ids = common_ids or set()
    h, w = img.shape[:2]
    for d in detections:
        pts = d.corners.astype(int)
        if d.tag_id in common_ids:
            color = (0, 0, 255)
            thickness = 3
        elif d.tag_id in floor_ids:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (255, 100, 0)
            thickness = 2

        cv2.polylines(ann, [pts], True, color, thickness)
        cx, cy = pts.mean(axis=0).astype(int)
        label = f"#{d.tag_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(ann, (cx-tw//2-2, cy-th-6), (cx+tw//2+2, cy-2), (0,0,0), -1)
        cv2.putText(ann, label, (cx-tw//2, cy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return ann

# ==============================================================
# 主分析
# ==============================================================

def main():
    print("[加载] 读取真实配置...")
    floor_tags = load_floor_tags()
    extrinsics = load_extrinsics()
    cam_config = load_camera_config()
    pairs = pair_snapshots()

    print(f"  地面 Tag: {len(floor_tags)} 个")
    print(f"  外参相机: {list(extrinsics.keys())}")
    print(f"  配对图像: {len(pairs)} 组\n")

    floor_ids = set(floor_tags.keys())

    # 从 cameras_config 提取各相机的内参
    cam_params = {}
    for cam_name, cfg in cam_config["cameras"].items():
        K_data = cfg["intrinsics"]["camera_matrix"]
        K = np.array(K_data, dtype=np.float64)
        dist = np.array(cfg["intrinsics"]["distortion_coefficients"], dtype=np.float64)
        res = tuple(cfg["resolution"])
        cam_params[cam_name] = {"K": K, "dist": dist, "res": res}
        print(f"  {cam_name}: {res[0]}x{res[1]}  fx={K[0,0]:.1f}")

    detector = Detector(families="tag36h11")

    # 逐组分析
    for ts in sorted(pairs.keys()):
        paths = pairs[ts]
        print(f"\n{'='*60}")
        print(f"【{ts}】")
        print(f"{'='*60}")

        img_picam = cv2.imread(paths["picam"])
        img_usb   = cv2.imread(paths["usb"])
        if img_picam is None or img_usb is None:
            continue

        # 检测（picam 和 usb）
        gray_p = cv2.cvtColor(img_picam, cv2.COLOR_BGR2GRAY)
        det_picam = detector.detect(gray_p)

        gray_u = cv2.cvtColor(img_usb, cv2.COLOR_BGR2GRAY)
        # USB 原图是 640x480，但配置说 2560x1440 — 实拍可能是低分辨率模式
        det_usb = detector.detect(gray_u)
        # CLAHE 增强
        if not det_usb:
            clahe = cv2.createCLAHE(2.0, (8,8))
            gray_u_enh = clahe.apply(gray_u)
            det_usb = detector.detect(gray_u_enh)
            if det_usb:
                print(f"  [USB CLAHE增强]")

        ids_p = set(d.tag_id for d in det_picam)
        ids_u = set(d.tag_id for d in det_usb)
        common = ids_p & ids_u

        # 摘要
        p_floor = ids_p & floor_ids
        u_floor = ids_u & floor_ids

        print(f"  piCam 检出: {len(det_picam):3d} tags  地面Tag: {sorted(p_floor)[:20]}{'...' if len(p_floor)>20 else ''}")
        print(f"  USB   检出: {len(det_usb):3d} tags  地面Tag: {sorted(u_floor)[:20]}{'...' if len(u_floor)>20 else ''}")
        if common:
            print(f"  ★ 共视 Tag: {sorted(common)} ({len(common)} 个)")
        else:
            print(f"  [无共视]")

        # 如果有共视 Tag，用外参验证几何一致性
        if common and "picam_1" in extrinsics and "usb_cam_1" in extrinsics:
            R_p, t_p = extrinsics["picam_1"]
            R_u, t_u = extrinsics["usb_cam_1"]
            # 用共视 Tag 的世界坐标反投到两相机，对比像素误差
            for tid in sorted(common)[:5]:
                wpt = floor_tags[tid]
                # 世界→piCam
                pc = R_p @ wpt + t_p.flatten()
                if pc[2] <= 0:
                    continue
                K_p = cam_params["picam_1"]["K"]
                pp = K_p @ (pc[:2] / pc[2])
                # 世界→USB
                uc = R_u @ wpt + t_u.flatten()
                if uc[2] <= 0:
                    continue
                K_u = cam_params["usb_cam_1"]["K"]
                pu = K_u @ (uc[:2] / uc[2])
                # 找实际检测的 Tag 中心
                for d in det_picam:
                    if d.tag_id == tid:
                        dp = d.center
                        break
                else:
                    dp = None
                for d in det_usb:
                    if d.tag_id == tid:
                        du = d.center
                        break
                else:
                    du = None
                if dp is not None:
                    e_p = np.linalg.norm(np.array(dp) - pp)
                    print(f"    Tag {tid}: piCam 检测({dp[0]:.0f},{dp[1]:.0f}) "
                          f"vs 外参投影({pp[0]:.0f},{pp[1]:.0f}) 误差={e_p:.1f}px")
                if du is not None:
                    e_u = np.linalg.norm(np.array(du) - pu)
                    print(f"          : USB  检测({du[0]:.0f},{du[1]:.0f}) "
                          f"vs 外参投影({pu[0]:.0f},{pu[1]:.0f}) 误差={e_u:.1f}px")

        # 并排显示
        ann_p = draw_tags(img_picam, det_picam, floor_ids, common)
        ann_u = draw_tags(img_usb, det_usb, floor_ids, common)

        h = max(ann_p.shape[0], ann_u.shape[0])
        rp = h / ann_p.shape[0]
        ru = h / ann_u.shape[0]
        p_small = cv2.resize(ann_p, (int(ann_p.shape[1]*rp), h))
        u_small = cv2.resize(ann_u, (int(ann_u.shape[1]*ru), h))

        combined = np.hstack([p_small, u_small])
        bar = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, f"piCam {ts}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(bar, f"USB {ts}", (p_small.shape[1]+10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        combined = np.vstack([bar, combined])

        out = f"snapshots/analysis_v2_{ts}.jpg"
        cv2.imwrite(out, combined)
        print(f"  [输出] {out}")

    print(f"\n{'='*60}")
    print("[完成]")

if __name__ == "__main__":
    main()
