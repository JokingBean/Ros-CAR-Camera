"""
双相机快照分析脚本 — ROS-Camera
================================
使用 snapshots/ 中的 piCam + USB 配对图像，
加载 config.yaml 的相机参数，检测 AprilTag 并并排展示。"""

import os
import sys
import cv2
import numpy as np
import yaml
from collections import defaultdict
from pupil_apriltags import Detector

# ==============================================================
# 加载配置
# ==============================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_floor_tags(path="floor_tags.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {int(k): np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
            for k, v in raw["tags"].items()}

# ==============================================================
# 图片配对
# ==============================================================

def pair_snapshots(snap_dir="snapshots"):
    """按时间戳配对 piCam 和 USB 快照。"""
    import re
    pairs = defaultdict(dict)
    for f in sorted(os.listdir(snap_dir)):
        if not f.endswith(".jpg"):
            continue
        # 从文件名提取时间戳: YYYYMMDD_HHMMSS
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
# Tag 检测
# ==============================================================

def detect_tags(image, detector=None):
    """检测一帧中的 AprilTag，返回 Detection 列表。
    对低分辨率图像自动上采样以提高检测率。"""
    if detector is None:
        detector = Detector(families="tag36h11")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 分辨率优化：低于 800px 宽时上采样 1.5x
    h, w = gray.shape
    if w < 800:
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    return detector.detect(gray)

def draw_detections(image, detections, floor_ids=None, highlight_ids=None):
    """在图像上绘制 Tag 检测结果。"""
    ann = image.copy()
    if floor_ids is None:
        floor_ids = set()
    if highlight_ids is None:
        highlight_ids = set()

    for d in detections:
        pts = d.corners.astype(int)
        # 地面 Tag 绿色，未知 Tag 蓝色，高亮（共同 Tag）红色
        if d.tag_id in highlight_ids:
            color = (0, 0, 255)          # 红色 = 两相机共视
        elif d.tag_id in floor_ids:
            color = (0, 255, 0)          # 绿色 = 已知地面 Tag
        else:
            color = (255, 0, 0)          # 蓝色 = 未知 Tag

        cv2.polylines(ann, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(ann, f"ID:{d.tag_id}",
                    (cx - 20, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 画角点号
        for i, p in enumerate(pts):
            cv2.circle(ann, tuple(p), 3, color, -1)

    return ann

# ==============================================================
# 并排可视化
# ==============================================================

def make_side_by_side(img_left, img_right,
                       label_left="piCam", label_right="USB",
                       pad=20):
    """两张图水平并排 + 标签。"""
    # 统一高度
    h = max(img_left.shape[0], img_right.shape[0])
    lw = int(h * img_left.shape[1] / img_left.shape[0])
    rw = int(h * img_right.shape[1] / img_right.shape[0])
    left = cv2.resize(img_left, (lw, h))
    right = cv2.resize(img_right, (rw, h))

    # 顶部加标签条
    bar_height = 40
    total_w = lw + pad + rw
    canvas = np.zeros((h + bar_height, total_w, 3), dtype=np.uint8)
    canvas[bar_height:, :lw] = left
    canvas[bar_height:, lw+pad:] = right

    # 标签
    cv2.putText(canvas, label_left, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, label_right, (lw + pad + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return canvas

# ==============================================================
# 主分析逻辑
# ==============================================================

def main():
    config = load_config()
    floor_tags = load_floor_tags()
    floor_ids = set(floor_tags.keys())
    pairs = pair_snapshots()

    if not pairs:
        print("[错误] snapshots/ 中未找到配对图片")
        sys.exit(1)

    print(f"[分析] 找到 {len(pairs)} 组配对")
    print(f"[分析] 已知地面 Tag: {sorted(floor_ids)} ({len(floor_ids)} 个)")
    print()

    detector = Detector(families="tag36h11")

    for ts in sorted(pairs.keys()):
        paths = pairs[ts]
        print(f"\n{'='*60}")
        print(f"时间戳: {ts}")
        print(f"{'='*60}")

        # 加载图片
        img_picam = cv2.imread(paths["picam"])
        img_usb   = cv2.imread(paths["usb"])

        if img_picam is None or img_usb is None:
            print("  [跳过] 图片缺失")
            continue

        # 检测 Tag
        det_picam = detect_tags(img_picam, detector)
        det_usb   = detect_tags(img_usb, detector)

        # USB 增强模式：如果标准检测结果为 0，尝试 CLAHE + 不同参数
        usb_enhanced = False
        if len(det_usb) == 0:
            gray_usb = cv2.cvtColor(img_usb, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_enh = clahe.apply(gray_usb)
            if gray_enh.shape[1] < 800:
                gray_enh = cv2.resize(gray_enh, None, fx=1.5, fy=1.5)
            det_usb_enh = detector.detect(gray_enh)
            if det_usb_enh:
                usb_enhanced = True
                det_usb = det_usb_enh
                print(f"  [USB增强] CLAHE+上采样后检出 {len(det_usb)} tags")

        ids_picam = set(d.tag_id for d in det_picam)
        ids_usb   = set(d.tag_id for d in det_usb)
        common    = ids_picam & ids_usb

        # 摘要
        print(f"  piCam ({img_picam.shape[1]}x{img_picam.shape[0]}):"
              f" {len(det_picam)} tags -> {sorted(ids_picam)}")
        print(f"  USB   ({img_usb.shape[1]}x{img_usb.shape[0]}):"
              f" {len(det_usb)} tags -> {sorted(ids_usb)}")
        if common:
            print(f"  >>> 共视 Tag: {sorted(common)} <<<")
        else:
            print(f"  [注意] 两相机无共视 Tag")

        # 标记哪些是已知地面 Tag
        floor_seen_picam = ids_picam & floor_ids
        floor_seen_usb   = ids_usb & floor_ids
        if floor_seen_picam:
            print(f"  piCam 地面Tag: {sorted(floor_seen_picam)}")
        if floor_seen_usb:
            print(f"  USB   地面Tag: {sorted(floor_seen_usb)}")

        # 详细：每个 Tag 的中心像素坐标
        for label, dets in [("piCam", det_picam), ("USB", det_usb)]:
            for d in dets:
                cx, cy = d.center
                is_floor = "G" if d.tag_id in floor_ids else " "
                is_common = "C" if d.tag_id in common else " "
                print(f"  [{label}] Tag ID:{d.tag_id:3d}  "
                      f"center=({cx:6.1f},{cy:6.1f})  "
                      f"decision={d.decision_margin:.1f} "
                      f"floor={is_floor} common={is_common}")

        # 可视化
        ann_picam = draw_detections(img_picam, det_picam,
                                    floor_ids=floor_ids,
                                    highlight_ids=common)
        ann_usb   = draw_detections(img_usb, det_usb,
                                    floor_ids=floor_ids,
                                    highlight_ids=common)

        usb_lbl = f"USB ({paths['usb']})"
        if usb_enhanced:
            usb_lbl += " [CLAHE+]"

        combined = make_side_by_side(ann_picam, ann_usb,
                                     label_left=f"piCam ({paths['picam']})",
                                     label_right=usb_lbl)

        # 保存结果到 snapshots/analysis_*.jpg
        out_path = f"snapshots/analysis_{ts}.jpg"
        cv2.imwrite(out_path, combined)
        print(f"  [输出] {out_path}")

    print(f"\n{'='*60}")
    print("[完成] 分析结果已保存至 snapshots/analysis_*.jpg")

if __name__ == "__main__":
    main()
