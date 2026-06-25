"""
三相机 BEV 俯视图融合 + Tag 精度分析
======================================
picam_1 (Pi) + usb_cam_1 (Pi) + usb_cam_2 (PC)
逐 Tag GSD 比较，标注最优相机。"""

import cv2, yaml, json
import numpy as np

# ==============================================================
# 配置
# ==============================================================
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = -0.5, 5.0
PPM = 200                     # px/m
M = 50
W = int((X_MAX - X_MIN) * PPM) + 2 * M
H = int((Y_MAX - Y_MIN) * PPM) + 2 * M

def w2p(x, y):
    return (M + int((x - X_MIN) * PPM), H - M - int((y - Y_MIN) * PPM))

# ==============================================================
# 加载
# ==============================================================
with open("extrinsics.yaml", "r") as f: ext = yaml.safe_load(f)

cameras = {
    "picam_1": {
        "K": np.array([[1064.8132,0,656.2857],[0,1056.9046,526.8922],[0,0,1]], dtype=np.float64),
        "R": np.array(ext["picam_1"]["R"]),
        "t": np.array(ext["picam_1"]["t"]).reshape(3,1),
        "img": cv2.imread("picam_calib.jpg"),
        "color": (255, 100, 60),   # blue-ish
        "label": "PiCam",
    },
    "usb_cam_1": {
        "K": np.array([[1610.2608,0,962.8233],[0,1599.8428,804.8184],[0,0,1]], dtype=np.float64),
        "R": np.array(ext["usb_cam_1"]["R"]),
        "t": np.array(ext["usb_cam_1"]["t"]).reshape(3,1),
        "img": cv2.imread("usb_calib.jpg"),
        "color": (60, 180, 255),   # orange-ish
        "label": "USB1",
    },
    "usb_cam_2": {
        "K": np.array([[1997.5587,0,1203.9179],[0,2004.3731,784.2230],[0,0,1]], dtype=np.float64),
        "R": np.array(ext["usb_cam_2"]["R"]),
        "t": np.array(ext["usb_cam_2"]["t"]).reshape(3,1),
        "img": cv2.imread("usb2_calib.jpg"),
        "color": (60, 255, 100),   # green-ish
        "label": "USB2",
    },
}

with open("floor_tags.yaml", "r", encoding="utf-8") as f:
    ft = yaml.safe_load(f)
floor_tags = {int(k): (v["x"], v["y"]) for k, v in ft["tags"].items()}

# ==============================================================
# BEV 投影
# ==============================================================
def project(x, y, z, K, R, t):
    P = np.array([[x],[y],[z]], dtype=np.float64)
    Pc = R @ P + t
    if Pc[2,0] <= 0: return None
    uv = K @ Pc
    return (uv[0,0]/uv[2,0], uv[1,0]/uv[2,0])

def point_visible(x, y, z, K, R, t, w, h):
    uv = project(x, y, z, K, R, t)
    if uv is None: return False
    return 0 <= uv[0] < w and 0 <= uv[1] < h

def gsd(x, y, z, K, R, t):
    P = np.array([[x],[y],[z]], dtype=np.float64)
    dist = np.linalg.norm(R @ P + t)
    return dist / ((K[0,0]+K[1,1])/2) * 1000

# ==============================================================
# 生成 BEV
# ==============================================================
bevs, masks = {}, {}
for name, cam in cameras.items():
    print(f"BEV: {name}...")
    img = cam["img"]
    h, w = img.shape[:2]
    bev = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    step = 1.0 / PPM
    for bv in range(H):
        yw = Y_MAX - (bv - M) * step
        for bu in range(W):
            xw = X_MIN + (bu - M) * step
            uv = project(xw, yw, 0.0, cam["K"], cam["R"], cam["t"])
            if uv is None: continue
            ui, vi = int(round(uv[0])), int(round(uv[1]))
            if 0 <= ui < w and 0 <= vi < h:
                bev[bv, bu] = img[vi, ui]
                mask[bv, bu] = 255
    bevs[name] = bev
    masks[name] = mask

# ==============================================================
# 融合
# ==============================================================
print("Fusing...")
fused = np.zeros_like(bevs["picam_1"])
count = np.zeros((H, W), dtype=np.float32)

for name in cameras:
    m = masks[name] > 0
    fused[m] = fused[m].astype(np.float32) + bevs[name][m].astype(np.float32)
    count[m] += 1.0
valid = count > 0
fused[valid] = (fused[valid] / count[valid, None]).astype(np.uint8)

# ==============================================================
# Tag 精度分析 + 标注
# ==============================================================
print("Analyzing tags...")
tag_data = []
for tid in sorted(floor_tags.keys()):
    tx, ty = floor_tags[tid]
    u, v = w2p(tx, ty)
    if not (0 <= u < W and 0 <= v < H): continue

    best, best_gsd, best_name = None, 999, ""
    visible = {}
    for name, cam in cameras.items():
        if point_visible(tx, ty, 0.0, cam["K"], cam["R"], cam["t"], cam["img"].shape[1], cam["img"].shape[0]):
            g = gsd(tx, ty, 0.0, cam["K"], cam["R"], cam["t"])
            visible[name] = g
            if g < best_gsd:
                best_gsd, best_name = g, name

    if not visible: continue

    # 颜色：最佳相机颜色
    if best_name == "picam_1":     dot = (255, 60, 30)
    elif best_name == "usb_cam_1": dot = (30, 130, 255)
    else:                          dot = (30, 220, 80)

    n_vis = len(visible)
    r = 4 + n_vis * 2   # 越多人看到圈越大
    cv2.circle(fused, (u, v), r, dot, -1)
    cv2.circle(fused, (u, v), r+1, (0,0,0), 1)
    if tid % 10 == 0:
        cv2.putText(fused, str(tid), (u+6, v+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    tag_data.append({
        "id": tid, "x": tx, "y": ty,
        "n_visible": n_vis,
        "best": best_name,
        "gsd_mm": {n: round(g,1) for n, g in visible.items()},
    })

# ==============================================================
# 相机位置
# ==============================================================
for name, cam in cameras.items():
    pos = (-cam["R"].T @ cam["t"]).flatten()
    pu, pv = w2p(pos[0], pos[1])
    cv2.circle(fused, (pu, pv), 16, (0,0,0), 2)
    cv2.circle(fused, (pu, pv), 14, cam["color"], -1)
    cv2.putText(fused, cam["label"], (pu+18, pv+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cam["color"], 2)

# ==============================================================
# 图例
# ==============================================================
lx, ly = W - 240, 40
cv2.rectangle(fused, (lx-5, ly-5), (lx+235, ly+120), (30,30,30), -1)
cv2.putText(fused, "TAG QUALITY (best GSD)", (lx, ly+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
for i, (name, cam) in enumerate(cameras.items()):
    cv2.circle(fused, (lx+14, ly+32+i*22), 5, cam["color"], -1)
    cv2.putText(fused, cam["label"], (lx+24, ly+36+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
cv2.circle(fused, (lx+14, ly+98), 4, (180,180,180), -1)
cv2.putText(fused, "3-cam overlap", (lx+24, ly+102), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)

# 比例尺
cv2.line(fused, (M, H-25), (M+PPM, H-25), (255,255,255), 4)
cv2.putText(fused, "1m", (M+10, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.putText(fused, f"3-Camera BEV  {PPM}px/m  {X_MIN}-{X_MAX}x{Y_MIN}-{Y_MAX}m",
            (M, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

cv2.imwrite("bev_3cam.jpg", fused)
print(f"bev_3cam.jpg saved ({W}x{H})")

# ==============================================================
# 统计
# ==============================================================
from collections import Counter
cnt = Counter(t["best"] for t in tag_data)
n_123 = sum(1 for t in tag_data if t["n_visible"] == 3)
n_12 = sum(1 for t in tag_data if t["n_visible"] == 2)

print(f"\nTags in view: {len(tag_data)} total")
print(f"  3-camera: {n_123}  |  2-camera: {n_12}  |  1-camera: {len(tag_data)-n_123-n_12}")
print(f"Best camera: PiCam={cnt['picam_1']}  USB1={cnt['usb_cam_1']}  USB2={cnt['usb_cam_2']}")

# 覆盖
for name in cameras:
    pct = (masks[name] > 0).sum() / (W * H) * 100
    print(f"  {name} coverage: {pct:.1f}%")

# ==============================================================
# HTML 报告
# ==============================================================
from datetime import datetime

rows = ""
for t in tag_data:
    best = t["best"]
    gsds = t["gsd_mm"]
    p_gsd = gsds.get("picam_1", "-")
    u1_gsd = gsds.get("usb_cam_1", "-")
    u2_gsd = gsds.get("usb_cam_2", "-")
    color = {"picam_1": "#ff5c3c", "usb_cam_1": "#3c82ff", "usb_cam_2": "#3cff50"}[best]
    rows += f'''<tr>
<td>{t["id"]}</td><td>{t["x"]:.2f}</td><td>{t["y"]:.2f}</td><td>{t["n_visible"]}</td>
<td>{p_gsd}</td><td>{u1_gsd}</td><td>{u2_gsd}</td>
<td style="color:{color};font-weight:bold">{best}</td>
</tr>\n'''

html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>3-Camera BEV Tag Analysis</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;margin:30px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#e94560}}h2{{background:#e0e0e0;color:#0f3460;padding:6px 14px;border-radius:4px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#16213e;border-radius:8px;padding:14px 20px;min-width:120px}}
.card h3{{margin:0 0 6px;font-size:12px;color:#888;text-transform:uppercase}}
.card .v{{font-size:24px;font-weight:bold}}
.picam{{color:#ff5c3c}}.usb1{{color:#3c82ff}}.usb2{{color:#3cff50}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th{{background:#0f3460;padding:8px 10px;text-align:left;position:sticky;top:0}}
td{{padding:5px 10px;border-bottom:1px solid #333}}
img{{max-width:100%;border-radius:8px;margin:16px 0}}
.foot{{color:#666;font-size:11px;margin-top:30px}}
</style></head><body>
<h1>3-Camera BEV Tag Precision Analysis</h1>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
<strong>Method:</strong> GSD (Ground Sampling Distance, mm/px) — smaller = better</p>

<div class="cards">
<div class="card"><h3>Tags in View</h3><div class="v">{len(tag_data)}</div></div>
<div class="card"><h3>3-Camera Tags</h3><div class="v">{n_123}</div></div>
<div class="card"><h3>PiCam Best</h3><div class="v picam">{cnt['picam_1']}</div></div>
<div class="card"><h3>USB1 Best</h3><div class="v usb1">{cnt['usb_cam_1']}</div></div>
<div class="card"><h3>USB2 Best</h3><div class="v usb2">{cnt['usb_cam_2']}</div></div>
</div>

<h2>BEV Image</h2>
<img src="bev_3cam.jpg" alt="3-Camera BEV">
<p>Dot color = best camera for that tag. Larger dot = more cameras see it.</p>

<h2>Tag-by-Tag GSD (mm/px)</h2>
<table>
<tr><th>ID</th><th>X</th><th>Y</th><th>#Cams</th><th>PiCam GSD</th><th>USB1 GSD</th><th>USB2 GSD</th><th>Best</th></tr>
{rows}
</table>
<div class="foot">ROS-Camera 3-Camera BEV Report — auto-generated</div>
</body></html>'''

with open("bev_3cam_report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("bev_3cam_report.html saved")
