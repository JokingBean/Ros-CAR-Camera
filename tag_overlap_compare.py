"""
Tag 重叠区对比图 — 逐 Tag 展示两相机画面差异
==============================================
对每个共视 Tag，从两相机原始图中裁出 Tag 区域并排对比，
标注 GSD 和精度优劣。"""

import cv2, yaml, json
import numpy as np
from pupil_apriltags import Detector

# ==============================================================
# 内参外参
# ==============================================================
K_p = np.array([[1064.8132, 0, 656.2857], [0, 1056.9046, 526.8922], [0, 0, 1]], dtype=np.float64)
K_u = np.array([[1610.2608, 0, 962.8233], [0, 1599.8428, 804.8184], [0, 0, 1]], dtype=np.float64)

with open("extrinsics.yaml", "r") as f:
    ext = yaml.safe_load(f)
R_p = np.array(ext["picam_1"]["R"]); t_p = np.array(ext["picam_1"]["t"]).reshape(3,1)
R_u = np.array(ext["usb_cam_1"]["R"]); t_u = np.array(ext["usb_cam_1"]["t"]).reshape(3,1)

with open("floor_tags.yaml", "r", encoding="utf-8") as f:
    ft = yaml.safe_load(f)
floor_tags = {int(k): (v["x"], v["y"], v["z"]) for k, v in ft["tags"].items()}

img_p = cv2.imread("picam_calib.jpg")
img_u = cv2.imread("usb_calib.jpg")

# ==============================================================
# 检测 Tag
# ==============================================================
clahe = cv2.createCLAHE(2.0, (8,8))
detector = Detector(families="tag36h11", quad_decimate=1.0)

gray_p = clahe.apply(cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY))
gray_u = clahe.apply(cv2.resize(cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5))

dets_p = {r.tag_id: r for r in detector.detect(gray_p)}
dets_u_raw = detector.detect(gray_u)
dets_u = {}
for r in dets_u_raw:
    r.corners *= 2.0; r.center = (r.center[0]*2.0, r.center[1]*2.0)
    dets_u[r.tag_id] = r

common = sorted(set(dets_p.keys()) & set(dets_u.keys()))
print(f"共视 Tag: {len(common)} 个 — {common}")

# ==============================================================
# GSD
# ==============================================================
def gsd_at_point(x, y, z, K, R, t):
    P = np.array([[x],[y],[z]], dtype=np.float64)
    dist = np.linalg.norm(R @ P + t)
    focal = (K[0,0] + K[1,1]) / 2.0
    return dist / focal * 1000.0   # mm/px

# ==============================================================
# 构建对比图
# ==============================================================
PATCH_SIZE = 80    # 每个 Tag patch 的边长
COLS = 5           # 每行放几个
ROWS = (len(common) + COLS - 1) // COLS
GAP = 4            # 间距
HEADER = 24        # 每行标题高度

cell_w = PATCH_SIZE * 2 + GAP   # piCam + USB 并排
cell_h = PATCH_SIZE + HEADER
canvas_w = cell_w * COLS + GAP * (COLS + 1)
canvas_h = cell_h * ROWS + GAP * (ROWS + 1) + 40

canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# 标题
cv2.putText(canvas, f"Tag Overlap Comparison — {len(common)} common tags  |  P=PiCam  U=USB  GSD=mm/px",
            (10, canvas_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

for idx, tid in enumerate(common):
    row, col = idx // COLS, idx % COLS
    x0 = GAP + col * cell_w
    y0 = GAP + row * cell_h

    # GSD
    tx, ty, tz = floor_tags[tid]
    gsd_p = gsd_at_point(tx, ty, tz, K_p, R_p, t_p)
    gsd_u = gsd_at_point(tx, ty, tz, K_u, R_u, t_u)
    better = "PiCam" if gsd_p <= gsd_u else "USB"
    diff_pct = abs(gsd_p - gsd_u) / min(gsd_p, gsd_u) * 100

    # 标题
    title_color = (80, 200, 255) if better == "USB" else (80, 150, 255)
    cv2.putText(canvas, f"#{tid} {better} wins",
                (x0+2, y0+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, title_color, 1)
    cv2.putText(canvas, f"P:{gsd_p:.1f} U:{gsd_u:.1f}",
                (x0+2, y0+HEADER-16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    # 分隔线
    cv2.line(canvas, (x0 + PATCH_SIZE, y0), (x0 + PATCH_SIZE, y0 + cell_h), (60,60,60), 1)

    # PiCam patch
    d_p = dets_p[tid]
    corners_p = d_p.corners.astype(int)
    cx_p, cy_p = int(d_p.center[0]), int(d_p.center[1])
    x1, y1 = max(0, cx_p - PATCH_SIZE//2), max(0, cy_p - PATCH_SIZE//2)
    x2, y2 = min(img_p.shape[1], x1 + PATCH_SIZE), min(img_p.shape[0], y1 + PATCH_SIZE)
    patch_p = img_p[y1:y2, x1:x2].copy()

    # 在 patch 上画 Tag 轮廓
    for i in range(4):
        pt1 = (corners_p[i][0] - x1, corners_p[i][1] - y1)
        pt2 = (corners_p[(i+1)%4][0] - x1, corners_p[(i+1)%4][1] - y1)
        cv2.line(patch_p, pt1, pt2, (0, 255, 0), 1)
    cv2.putText(patch_p, f"P #{tid}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)

    # 填充到固定大小
    pw, ph = PATCH_SIZE, PATCH_SIZE
    if patch_p.shape[1] < pw or patch_p.shape[0] < ph:
        padded = np.zeros((ph, pw, 3), dtype=np.uint8)
        padded[:patch_p.shape[0], :patch_p.shape[1]] = patch_p[:ph, :pw]
        patch_p = padded
    else:
        patch_p = patch_p[:ph, :pw]

    canvas[y0+HEADER:y0+HEADER+ph, x0:x0+pw] = patch_p

    # USB patch
    d_u = dets_u[tid]
    corners_u = d_u.corners.astype(int)
    cx_u, cy_u = int(d_u.center[0]), int(d_u.center[1])
    x1u, y1u = max(0, cx_u - PATCH_SIZE//2), max(0, cy_u - PATCH_SIZE//2)
    x2u, y2u = min(img_u.shape[1], x1u + PATCH_SIZE), min(img_u.shape[0], y1u + PATCH_SIZE)
    patch_u = img_u[y1u:y2u, x1u:x2u].copy()

    for i in range(4):
        pt1 = (corners_u[i][0] - x1u, corners_u[i][1] - y1u)
        pt2 = (corners_u[(i+1)%4][0] - x1u, corners_u[(i+1)%4][1] - y1u)
        cv2.line(patch_u, pt1, pt2, (0, 255, 255), 1)
    cv2.putText(patch_u, f"U #{tid}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

    if patch_u.shape[1] < pw or patch_u.shape[0] < ph:
        padded = np.zeros((ph, pw, 3), dtype=np.uint8)
        padded[:patch_u.shape[0], :patch_u.shape[1]] = patch_u[:ph, :pw]
        patch_u = padded
    else:
        patch_u = patch_u[:ph, :pw]

    canvas[y0+HEADER:y0+HEADER+ph, x0+pw+GAP:x0+pw+GAP+pw] = patch_u

# 图例
ly = canvas_h - 30
cv2.putText(canvas, "Green=PiCam  Yellow=USB  Thicker outline = better GSD", (10, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

cv2.imwrite("tag_comparison.jpg", canvas)
print(f"tag_comparison.jpg 已保存 ({canvas.shape[1]}x{canvas.shape[0]})")
print(f"共 {len(common)} 个 Tag，PiCam 更优: {sum(1 for tid in common if gsd_at_point(*floor_tags[tid], K_p, R_p, t_p) <= gsd_at_point(*floor_tags[tid], K_u, R_u, t_u))} 个")
print(f"共 {len(common)} 个 Tag，USB 更优:   {sum(1 for tid in common if gsd_at_point(*floor_tags[tid], K_u, R_u, t_u) < gsd_at_point(*floor_tags[tid], K_p, R_p, t_p))} 个")
