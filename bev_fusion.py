"""
双相机 BEV 俯视图融合 — Tag 点位精度分析
==========================================
在每个地面 Tag 位置计算两相机的 GSD（地面采样间距），
标注哪个相机更精确，误差差多少。输出 HTML 报告。"""

import cv2, yaml, numpy as np
from datetime import datetime

# ==============================================================
# 配置
# ==============================================================
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = 0.0, 5.0
PIXELS_PER_METER = 200
MARGIN = 50
OUT_W = int((X_MAX - X_MIN) * PIXELS_PER_METER) + 2 * MARGIN
OUT_H = int((Y_MAX - Y_MIN) * PIXELS_PER_METER) + 2 * MARGIN

def w2p(x, y):
    u = MARGIN + int((x - X_MIN) * PIXELS_PER_METER)
    v = OUT_H - MARGIN - int((y - Y_MIN) * PIXELS_PER_METER)
    return u, v

# ==============================================================
# 加载数据
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
# GSD 计算
# ==============================================================
def gsd_at_point(x, y, z, K, R, t):
    P = np.array([[x],[y],[z]], dtype=np.float64)
    P_c = R @ P + t
    dist = np.linalg.norm(P_c)
    focal = (K[0,0] + K[1,1]) / 2.0
    return dist / focal * 1000.0   # mm/px

def point_in_image(x, y, z, K, R, t, w, h):
    """点是否在相机视野内"""
    P = np.array([[x],[y],[z]], dtype=np.float64)
    P_c = R @ P + t
    if P_c[2,0] <= 0: return False
    uv = K @ P_c
    u, v = uv[0,0]/uv[2,0], uv[1,0]/uv[2,0]
    return 0 <= u < w and 0 <= v < h

# ==============================================================
# 逐 Tag 分析
# ==============================================================
hp, wp = img_p.shape[:2]
hu, wu = img_u.shape[:2]

tag_analysis = []
for tid in sorted(floor_tags.keys()):
    tx, ty, tz = floor_tags[tid]
    in_p = point_in_image(tx, ty, tz, K_p, R_p, t_p, wp, hp)
    in_u = point_in_image(tx, ty, tz, K_u, R_u, t_u, wu, hu)
    if not in_p and not in_u:
        continue

    gsd_p = gsd_at_point(tx, ty, tz, K_p, R_p, t_p) if in_p else None
    gsd_u = gsd_at_point(tx, ty, tz, K_u, R_u, t_u) if in_u else None

    # 判断优劣
    if gsd_p is not None and gsd_u is not None:
        if gsd_p <= gsd_u:
            better = "PiCam"
            diff_pct = (gsd_u - gsd_p) / gsd_p * 100
        else:
            better = "USB"
            diff_pct = (gsd_p - gsd_u) / gsd_u * 100
        ratio = max(gsd_p, gsd_u) / min(gsd_p, gsd_u)
    elif gsd_p is not None:
        better = "PiCam"
        diff_pct = None; ratio = None
    else:
        better = "USB"
        diff_pct = None; ratio = None

    tag_analysis.append({
        "id": tid,
        "x": tx, "y": ty,
        "in_picam": in_p, "in_usb": in_u,
        "gsd_picam": round(gsd_p, 2) if gsd_p else None,
        "gsd_usb": round(gsd_u, 2) if gsd_u else None,
        "better": better,
        "diff_pct": round(diff_pct, 1) if diff_pct else None,
        "ratio": round(ratio, 2) if ratio else None,
    })

# ==============================================================
# BEV 投影
# ==============================================================
def project_world_to_image(x, y, z, K, R, t):
    P_w = np.array([[x],[y],[z]], dtype=np.float64)
    P_c = R @ P_w + t
    if P_c[2,0] <= 0: return None
    uv = K @ P_c
    return (uv[0,0]/uv[2,0], uv[1,0]/uv[2,0])

def make_bev(img, K, R, t):
    bev = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    mask = np.zeros((OUT_H, OUT_W), dtype=np.uint8)
    h, w = img.shape[:2]
    step = 1.0 / PIXELS_PER_METER
    for bv in range(OUT_H):
        y_w = Y_MAX - (bv - MARGIN) * step
        for bu in range(OUT_W):
            x_w = X_MIN + (bu - MARGIN) * step
            uv = project_world_to_image(x_w, y_w, 0.0, K, R, t)
            if uv is None: continue
            ui, vi = int(round(uv[0])), int(round(uv[1]))
            if 0 <= ui < w and 0 <= vi < h:
                bev[bv, bu] = img[vi, ui]
                mask[bv, bu] = 255
    return bev, mask

print("生成 BEV...")
bev_p, mask_p = make_bev(img_p, K_p, R_p, t_p)
bev_u, mask_u = make_bev(img_u, K_u, R_u, t_u)

# 融合
bev_merged = np.zeros_like(bev_p)
bev_merged[mask_p > 0] = bev_p[mask_p > 0]
bev_merged[mask_u > 0] = bev_u[mask_u > 0]
overlap = (mask_p > 0) & (mask_u > 0)
if overlap.any():
    bev_merged[overlap] = ((bev_p[overlap].astype(np.float32) +
                            bev_u[overlap].astype(np.float32))/2).astype(np.uint8)

# ==============================================================
# 标注 Tag——只标 Tag 位置，颜色表示精度优劣
# ==============================================================
for ta in tag_analysis:
    u, v = w2p(ta["x"], ta["y"])
    if not (0 <= u < OUT_W and 0 <= v < OUT_H):
        continue

    # 颜色：PiCam 更好=蓝，USB 更好=红，单相机=灰
    if ta["better"] == "PiCam":
        if ta["in_usb"]:
            color = (255, 80, 40)    # 红蓝混合→PiCam 胜出（蓝底红色点）
            dot_color = (255, 140, 40)
        else:
            dot_color = (255, 200, 100)
    else:
        if ta["in_picam"]:
            dot_color = (40, 140, 255)  # USB 胜出（红底蓝点）
        else:
            dot_color = (100, 200, 255)

    cv2.circle(bev_merged, (u, v), 5, dot_color, -1)
    cv2.circle(bev_merged, (u, v), 6, (0, 0, 0), 1)

    if ta["better"] == "PiCam" and ta["in_usb"]:
        cv2.circle(bev_merged, (u, v), 10, (255, 80, 40), 1)  # 外圈=胜者
    elif ta["better"] == "USB" and ta["in_picam"]:
        cv2.circle(bev_merged, (u, v), 10, (40, 80, 255), 1)

    if ta["id"] % 10 == 0:
        cv2.putText(bev_merged, str(ta["id"]), (u+8, v+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

# 相机位置
cam_p = (-R_p.T @ t_p).flatten()
cam_u = (-R_u.T @ t_u).flatten()
for label, pos, col in [("PiCam", cam_p, (255,80,80)), ("USB", cam_u, (80,80,255))]:
    pu, pv = w2p(pos[0], pos[1])
    cv2.circle(bev_merged, (pu, pv), 15, col, -1)
    cv2.circle(bev_merged, (pu, pv), 17, (0,0,0), 2)
    cv2.putText(bev_merged, label, (pu+20, pv+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

# 图例
lx, ly = OUT_W - 230, 40
cv2.putText(bev_merged, "TAG GSD QUALITY", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.circle(bev_merged, (lx+15, ly+22), 5, (40, 140, 255), -1)
cv2.putText(bev_merged, "USB better (red ring)", (lx+28, ly+26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
cv2.circle(bev_merged, (lx+15, ly+42), 5, (255, 140, 40), -1)
cv2.putText(bev_merged, "PiCam better (blue ring)", (lx+28, ly+46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
cv2.circle(bev_merged, (lx+15, ly+62), 5, (255, 200, 100), -1)
cv2.putText(bev_merged, "PiCam only", (lx+28, ly+66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
cv2.circle(bev_merged, (lx+15, ly+82), 5, (100, 200, 255), -1)
cv2.putText(bev_merged, "USB only", (lx+28, ly+86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

# 比例尺
bar_y = OUT_H - 25
cv2.line(bev_merged, (MARGIN, bar_y), (MARGIN+PIXELS_PER_METER, bar_y), (255,255,255), 4)
cv2.putText(bev_merged, "1m", (MARGIN+10, bar_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.putText(bev_merged, f"BEV {X_MIN}-{X_MAX}m x {Y_MIN}-{Y_MAX}m  {PIXELS_PER_METER}px/m",
            (MARGIN, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

cv2.imwrite("bev_fused.jpg", bev_merged)
print("bev_fused.jpg 已保存")

# ==============================================================
# HTML 报告
# ==============================================================
n_p_better = sum(1 for t in tag_analysis if t["better"] == "PiCam" and t["in_usb"])
n_u_better = sum(1 for t in tag_analysis if t["better"] == "USB" and t["in_picam"])
n_p_only   = sum(1 for t in tag_analysis if t["in_picam"] and not t["in_usb"])
n_u_only   = sum(1 for t in tag_analysis if t["in_usb"] and not t["in_picam"])

# 重叠区 GSD 统计
overlap_gsd_p = [t["gsd_picam"] for t in tag_analysis if t["gsd_picam"] and t["gsd_usb"]]
overlap_gsd_u = [t["gsd_usb"] for t in tag_analysis if t["gsd_picam"] and t["gsd_usb"]]

html_rows = ""
for ta in tag_analysis:
    gsd_p_str = f'{ta["gsd_picam"]:.1f}' if ta["gsd_picam"] else "-"
    gsd_u_str = f'{ta["gsd_usb"]:.1f}' if ta["gsd_usb"] else "-"

    if ta["better"] == "PiCam" and ta["in_usb"]:
        better_str = f'<span style="color:#d44">PiCam</span>'
        diff_str = f'USB 差 {ta["diff_pct"]:.0f}%' if ta["diff_pct"] else "-"
        cls = "picam-win"
    elif ta["better"] == "USB" and ta["in_picam"]:
        better_str = f'<span style="color:#44d">USB</span>'
        diff_str = f'PiCam 差 {ta["diff_pct"]:.0f}%' if ta["diff_pct"] else "-"
        cls = "usb-win"
    elif ta["in_picam"]:
        better_str = '<span style="color:#888">仅PiCam</span>'
        diff_str = "-"
        cls = "picam-only"
    else:
        better_str = '<span style="color:#888">仅USB</span>'
        diff_str = "-"
        cls = "usb-only"

    html_rows += f'''<tr class="{cls}">
<td>{ta["id"]}</td><td>{ta["x"]:.2f}</td><td>{ta["y"]:.2f}</td>
<td>{gsd_p_str}</td><td>{gsd_u_str}</td>
<td>{better_str}</td><td>{diff_str}</td>
</tr>\n'''

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>双相机 Tag 精度分析报告</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #e94560; }}
h2 {{ color: #0f3460; background: #e0e0e0; padding: 8px 16px; border-radius: 4px; }}
.summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
.card {{ background: #16213e; border-radius: 8px; padding: 16px 24px; min-width: 150px; }}
.card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #888; text-transform: uppercase; }}
.card .value {{ font-size: 28px; font-weight: bold; }}
.picam {{ color: #ff8c5a; }}
.usb {{ color: #5a8cff; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
th {{ background: #0f3460; color: #e0e0e0; padding: 10px 12px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #333; }}
tr.picam-win {{ background: #2d1a1a; }}
tr.usb-win {{ background: #1a1a2d; }}
tr.picam-only {{ background: #1f1f1f; color: #888; }}
tr.usb-only {{ background: #1f1f1f; color: #888; }}
img {{ max-width: 100%; border-radius: 8px; margin: 20px 0; }}
.footer {{ color: #666; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>

<h1>双相机 Tag 点位精度分析报告</h1>
<p><strong>日期:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
<strong>方法:</strong> GSD (Ground Sampling Distance) 比较</p>

<div style="background:#1a2a1a; padding:12px 20px; border-radius:8px; margin:16px 0;">
<strong>什么是 GSD？</strong><br>
GSD = Ground Sampling Distance = 地面采样间距，单位 <strong>mm/px</strong>。<br>
它表示地面上一个像素覆盖多少毫米。<br>
<strong>GSD 越小 = 像素越密 = Tag 角点定位越精确。</strong><br>
例如：GSD=2.4mm 意味着 Tag 上一个像素误差对应地面 2.4mm 误差；GSD=3.7mm 则对应 3.7mm。
</div>

<h2>概览</h2>
<div class="summary">
<div class="card"><h3>可观测 Tag 总数</h3><div class="value" style="color:#fff">{len(tag_analysis)}</div></div>
<div class="card"><h3>PiCam 独有</h3><div class="value picam">{n_p_only}</div></div>
<div class="card"><h3>USB 独有</h3><div class="value usb">{n_u_only}</div></div>
<div class="card"><h3>重叠区 PiCam 胜</h3><div class="value picam">{n_p_better}</div></div>
<div class="card"><h3>重叠区 USB 胜</h3><div class="value usb">{n_u_better}</div></div>
<div class="card"><h3>重叠区 USB 胜率</h3><div class="value usb">{n_u_better}/{n_p_better+n_u_better}</div></div>
</div>

<h2>重叠区 GSD 统计 (mm/px)</h2>
<div class="summary">
<div class="card"><h3>PiCam 平均 GSD</h3><div class="value picam">{np.mean(overlap_gsd_p):.1f}</div></div>
<div class="card"><h3>USB 平均 GSD</h3><div class="value usb">{np.mean(overlap_gsd_u):.1f}</div></div>
<div class="card"><h3>USB 平均优势</h3><div class="value usb">{np.mean(overlap_gsd_p)/np.mean(overlap_gsd_u)*100-100:.1f}%</div></div>
</div>
<p>重叠区内 USB 的 GSD 平均比 PiCam 精细 <strong>{np.mean(overlap_gsd_p)/np.mean(overlap_gsd_u)*100-100:.1f}%</strong>，因为 USB 分辨率 2048×1536 高于 PiCam 的 1332×990。</p>

<h2>俯视图 (Tag 点位精度标注)</h2>
<img src="bev_fused.jpg" alt="BEV Fusion">
<p>
<span style="color:#ff8c5a">● 蓝圈</span> = PiCam GSD 更小（更精确）&nbsp;|&nbsp;
<span style="color:#5a8cff">● 红圈</span> = USB GSD 更小（更精确）&nbsp;|&nbsp;
<span>● 灰点</span> = 仅单相机可见
</p>

<h2>逐 Tag 详细数据</h2>
<p>GSD = 地面采样间距 (mm/px)，越小越好。差值 = 较差相机的 GSD 比较优相机大多少。</p>
<table>
<tr><th>Tag ID</th><th>X (m)</th><th>Y (m)</th><th>PiCam GSD</th><th>USB GSD</th><th>更精确</th><th>差值</th></tr>
{html_rows}
</table>

<div class="footer">
报告自动生成 &mdash; ROS-Camera 双相机追踪系统
</div>

</body>
</html>'''

with open("bev_report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("bev_report.html 已保存")
print(f"\n重叠区: PiCam 胜 {n_p_better} vs USB 胜 {n_u_better}")
print(f"USB 平均 GSD 优势: {np.mean(overlap_gsd_p)/np.mean(overlap_gsd_u)*100-100:.1f}%")
