"""
bev_generic.py — 通用 N 相机 BEV 俯视图融合 + Tag 精度分析
============================================================
支持任意数量相机（≥1），从配置文件动态加载参数。

两种模式:
  - offline: 从本地图片文件读取（指定图片路径映射）
  - live:    从相机实时捕获

用法:
  # CLI: 从 config.yaml 加载所有相机，离线模式（需指定图片）
  python bev_generic.py --offline --image picam_1=picam.jpg,usb_cam_1=usb1.jpg

  # CLI: 实时捕获
  python bev_generic.py --live --cameras picam_1,usb_cam_1,usb_cam_2

  # Python API
  from bev_generic import BevGenerator
  gen = BevGenerator()
  fused, tag_data = gen.run(camera_names=["picam_1", "usb_cam_1"])
  gen.save_report(fused, tag_data, camera_names=["picam_1", "usb_cam_1"])
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from collections import Counter


# ==============================================================
# 默认 BEV 投影参数
# ==============================================================
DEFAULT_X_MIN, DEFAULT_X_MAX = 0.0, 4.5
DEFAULT_Y_MIN, DEFAULT_Y_MAX = -0.5, 5.0
DEFAULT_PPM = 200          # pixels per meter
DEFAULT_MARGIN = 50


# ==============================================================
# 工具函数
# ==============================================================

def _w2p(x, y, x_min, x_max, y_min, y_max, ppm, margin, out_h):
    """世界坐标 → BEV 像素坐标。"""
    u = margin + int((x - x_min) * ppm)
    v = out_h - margin - int((y - y_min) * ppm)
    return u, v


def _project_world_to_image(x, y, z, K, R, t):
    """世界坐标点投影到图像平面。返回 (u, v) 或 None。"""
    P_w = np.array([[x], [y], [z]], dtype=np.float64)
    P_c = R @ P_w + t
    if P_c[2, 0] <= 0:
        return None
    uv = K @ P_c
    return (uv[0, 0] / uv[2, 0], uv[1, 0] / uv[2, 0])


def _gsd_at_point(x, y, z, K, R, t):
    """计算世界坐标某点的 GSD (mm/px)，越小越精细。"""
    P = np.array([[x], [y], [z]], dtype=np.float64)
    P_c = R @ P + t
    dist = np.linalg.norm(P_c)
    focal = (K[0, 0] + K[1, 1]) / 2.0
    return dist / focal * 1000.0


def _point_visible(x, y, z, K, R, t, img_w, img_h):
    """判断世界坐标点是否在相机视野内。"""
    uv = _project_world_to_image(x, y, z, K, R, t)
    if uv is None:
        return False
    return 0 <= uv[0] < img_w and 0 <= uv[1] < img_h


# ==============================================================
# BevGenerator 主类
# ==============================================================

class BevGenerator:
    """通用 N 相机 BEV 俯视图生成器。"""

    def __init__(self,
                 config_path="config.yaml",
                 extrinsics_path="extrinsics.yaml",
                 floor_tags_path="floor_tags.yaml",
                 x_min=DEFAULT_X_MIN, x_max=DEFAULT_X_MAX,
                 y_min=DEFAULT_Y_MIN, y_max=DEFAULT_Y_MAX,
                 ppm=DEFAULT_PPM, margin=DEFAULT_MARGIN):
        self.config_path = config_path
        self.extrinsics_path = extrinsics_path
        self.floor_tags_path = floor_tags_path
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.ppm = ppm
        self.margin = margin
        self.out_w = int((x_max - x_min) * ppm) + 2 * margin
        self.out_h = int((y_max - y_min) * ppm) + 2 * margin

        # 懒加载
        self._config = None
        self._extrinsics = None
        self._floor_tags = None
        self._all_camera_params = None

    # ----------------------------------------------------------
    # 配置加载
    # ----------------------------------------------------------

    def _load_config(self):
        """加载 config.yaml。"""
        if self._config is not None:
            return self._config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        return self._config

    def _load_extrinsics(self):
        """加载 extrinsics.yaml。"""
        if self._extrinsics is not None:
            return self._extrinsics
        with open(self.extrinsics_path, "r", encoding="utf-8") as f:
            self._extrinsics = yaml.safe_load(f)
        return self._extrinsics

    def _load_floor_tags(self):
        """加载 floor_tags.yaml。"""
        if self._floor_tags is not None:
            return self._floor_tags
        with open(self.floor_tags_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._floor_tags = {int(k): (v["x"], v["y"], v["z"])
                            for k, v in raw["tags"].items()}
        return self._floor_tags

    def _load_all_camera_params(self):
        """加载所有相机的内参+外参。返回 dict[name -> params]。"""
        if self._all_camera_params is not None:
            return self._all_camera_params

        config = self._load_config()
        ext = self._load_extrinsics()
        params = {}

        for cam_cfg in config["cameras"]:
            name = cam_cfg["name"]
            # 内参
            cm = cam_cfg["camera_matrix"]
            K = np.array([[cm["fx"], 0, cm["cx"]],
                          [0, cm["fy"], cm["cy"]],
                          [0, 0, 1]], dtype=np.float64)
            dist = np.array(cam_cfg["dist_coeffs"], dtype=np.float64)

            # 外参
            if name in ext:
                R = np.array(ext[name]["R"], dtype=np.float64)
                t = np.array(ext[name]["t"], dtype=np.float64).reshape(3, 1)
            else:
                print(f"[警告] {name} 没有外参数据，跳过")
                continue

            # 颜色（用 name hash 生成稳定色）
            color_bgr = self._name_to_color(name)
            params[name] = {
                "K": K,
                "dist": dist,
                "R": R,
                "t": t,
                "color": color_bgr,
                "type": cam_cfg.get("type", "usb"),
                "resolution": cam_cfg.get("resolution", [640, 480]),
            }

        self._all_camera_params = params
        return params

    @staticmethod
    def _name_to_color(name):
        """根据相机序号生成稳定颜色。"""
        colors = {
            "usb1": (255, 100, 60),
            "usb2": (60, 255, 100),
            "usb3": (60, 180, 255),
            "picam_1": (255, 100, 60),
            "usb_cam_1": (60, 180, 255),
            "usb_cam_2": (60, 255, 100),
        }
        if name in colors:
            return colors[name]
        import hashlib
        h = hashlib.md5(name.encode()).hexdigest()
        return (int(h[0:2], 16) % 200 + 55, int(h[2:4], 16) % 200 + 55, int(h[4:6], 16) % 200 + 55)

    @staticmethod
    def _name_to_label(name):
        """相机名 → 显示标签。"""
        labels = {
            "usb1": "USB1", "usb2": "USB2", "usb3": "USB3",
            "picam_1": "PiCam", "usb_cam_1": "USB1", "usb_cam_2": "USB2",
        }
        return labels.get(name, name)

    # ----------------------------------------------------------
    # 获取可用相机列表
    # ----------------------------------------------------------

    def get_available_cameras(self):
        """返回所有可用的相机名列表。"""
        params = self._load_all_camera_params()
        return list(params.keys())

    def get_camera_params(self, camera_names=None):
        """获取指定相机的参数。camera_names=None 返回全部。"""
        all_params = self._load_all_camera_params()
        if camera_names is None:
            return all_params
        return {n: all_params[n] for n in camera_names if n in all_params}

    # ----------------------------------------------------------
    # 图像获取
    # ----------------------------------------------------------

    def capture_images(self, camera_names):
        """实时捕获相机图像。返回 dict[name -> image(BGR)]。"""
        from camera_reader import CameraReader, open_all_cameras

        config = self._load_config()
        # 只开需要的相机
        selected_cfgs = [c for c in config["cameras"] if c["name"] in camera_names]
        if not selected_cfgs:
            raise ValueError(f"没有找到指定的相机: {camera_names}")

        readers = open_all_cameras({"cameras": selected_cfgs})
        import time
        time.sleep(0.5)  # 等待稳定

        images = {}
        for r in readers:
            for _ in range(3):  # 丢弃前几帧
                r.read()
            frame = r.read()
            if frame is not None:
                images[r.name] = frame
                print(f"  [{r.name}] 已捕获 {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  [{r.name}] 捕获失败")
            r.release()

        return images

    def load_images_from_files(self, file_map):
        """从文件加载图片。file_map: dict[name -> file_path]。
        返回 dict[name -> image(BGR)]。"""
        images = {}
        for name, path in file_map.items():
            img = cv2.imread(path)
            if img is not None:
                images[name] = img
                print(f"  [{name}] 已加载 {path} ({img.shape[1]}x{img.shape[0]})")
            else:
                print(f"  [{name}] 加载失败: {path}")
        return images

    # ----------------------------------------------------------
    # 单相机 BEV 投影
    # ----------------------------------------------------------

    def _make_bev(self, img, K, R, t):
        """将单张图像投影到 BEV 平面（使用内参+外参）。
        返回 (bev, mask)，bev 为 (H,W,3) BGR，mask 为 (H,W) uint8。"""
        h_img, w_img = img.shape[:2]
        bev = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        mask = np.zeros((self.out_h, self.out_w), dtype=np.uint8)
        step = 1.0 / self.ppm

        for bv in range(self.out_h):
            y_w = self.y_max - (bv - self.margin) * step
            for bu in range(self.out_w):
                x_w = self.x_min + (bu - self.margin) * step
                uv = _project_world_to_image(x_w, y_w, 0.0, K, R, t)
                if uv is None:
                    continue
                ui, vi = int(round(uv[0])), int(round(uv[1]))
                if 0 <= ui < w_img and 0 <= vi < h_img:
                    bev[bv, bu] = img[vi, ui]
                    mask[bv, bu] = 255
        return bev, mask

    def _make_bev_undistorted(self, img, K, dist, R, t):
        """去畸变后再投影 BEV，消除边缘拉伸和融合错位。"""
        h_img, w_img = img.shape[:2]

        # 去畸变映射
        if dist is not None and np.any(dist != 0):
            # 对齐 dist 维度
            d = dist.reshape(1, -1) if dist.ndim == 1 else dist
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, d, (w_img, h_img), 0, (w_img, h_img))
            map1, map2 = cv2.initUndistortRectifyMap(K, d, None, new_K,
                                                      (w_img, h_img), cv2.CV_32FC1)
            img_undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            K_use = new_K
        else:
            img_undist = img
            K_use = K

        return self._make_bev(img_undist, K_use, R, t)

    def _make_bev_from_homography(self, img, H):
        """用 homography 直接将世界坐标映射到图像（不依赖内参外参）。
        H: 3x3 矩阵，world_xy -> image_uv。
        返回 (bev, mask)。"""
        h_img, w_img = img.shape[:2]
        bev = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        mask = np.zeros((self.out_h, self.out_w), dtype=np.uint8)
        step = 1.0 / self.ppm

        # 用 H 批量计算世界坐标 -> 图像坐标
        # 构建世界坐标网格
        xs = np.arange(self.out_w)  # pixel x in BEV
        ys = np.arange(self.out_h)  # pixel y in BEV

        for bv in range(self.out_h):
            y_w = self.y_max - (bv - self.margin) * step
            # 整行一起算
            x_w_row = self.x_min + (xs - self.margin) * step  # shape (out_w,)
            # 世界坐标: (x_w_row, y_w, 0) -> 齐次坐标
            world_h = np.column_stack([x_w_row, np.full_like(x_w_row, y_w), np.ones_like(x_w_row)])  # (out_w, 3)
            uv_h = H @ world_h.T  # (3, out_w)
            u = (uv_h[0] / uv_h[2]).astype(int)  # (out_w,)
            v = (uv_h[1] / uv_h[2]).astype(int)  # (out_w,)

            valid = (0 <= u) & (u < w_img) & (0 <= v) & (v < h_img)
            bev[bv, valid] = img[v[valid], u[valid]]
            mask[bv, valid] = 255

        return bev, mask

    # ----------------------------------------------------------
    # 融合
    # ----------------------------------------------------------

    def _fuse_bevs(self, bevs, masks, homographies=None):
        """融合多相机 BEV。如果有 homography，用最近相机加权；否则平均。
        
        homographies: dict[name -> H] 可选，用于计算每像素的相机近邻权重。
        """
        if not bevs:
            return None, None

        fused = np.zeros_like(next(iter(bevs.values())), dtype=np.float32)
        count = np.zeros((self.out_h, self.out_w), dtype=np.float32)
        weights_sum = np.zeros((self.out_h, self.out_w), dtype=np.float32)

        names = list(bevs.keys())
        n_cams = len(names)

        if homographies and n_cams >= 2:
            # 用 homography 逆映射计算每台相机在 BEV 中的"中心"（相机主点对应的世界坐标）
            cam_centers = {}
            for name in names:
                H = homographies.get(name)
                if H is not None:
                    try:
                        H_inv = np.linalg.inv(H)
                        # 图像中心点
                        img = next(iter(bevs.values()))  # just for reference
                        # Actually we need the image dimensions. Let's use a standard approach:
                        # The camera's "nadir" in world = H_inv @ (cx, cy, 1)
                        # We don't have cx,cy here, but we can approximate the center as 
                        # the point that maps to the center of the visible region in the BEV mask.
                        # Simplest: use the mean of the mask as the camera "center"
                        m = masks[name]
                        ys, xs = np.where(m > 0)
                        if len(xs) > 0:
                            cam_centers[name] = np.array([xs.mean(), ys.mean()])
                    except:
                        pass

            if len(cam_centers) >= 2:
                # Per-pixel: weight by inverse distance to camera center
                # Pre-compute distance maps
                dist_maps = {}
                for name, center in cam_centers.items():
                    cx, cy = center
                    y_grid, x_grid = np.ogrid[:self.out_h, :self.out_w]
                    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                    # Normalize: max distance ~ diagonal of BEV
                    max_dist = np.sqrt(self.out_w ** 2 + self.out_h ** 2)
                    weight = 1.0 / (1.0 + dist / (max_dist / 4))  # 衰减
                    dist_maps[name] = weight

                for name in names:
                    m = masks[name] > 0
                    w = dist_maps.get(name, np.ones_like(count))
                    fused[m] += bevs[name][m].astype(np.float32) * w[m, None]
                    weights_sum[m] += w[m]
                    count[m] += 1.0

                valid = weights_sum > 0
                fused[valid] = (fused[valid] / weights_sum[valid, None]).astype(np.uint8)
                return fused.astype(np.uint8), count
            else:
                # Fall through to simple average
                pass

        # Simple average (original behavior)
        for name in names:
            m = masks[name] > 0
            fused[m] += bevs[name][m].astype(np.float32)
            count[m] += 1.0

        valid = count > 0
        fused[valid] = (fused[valid] / count[valid, None]).astype(np.uint8)
        return fused.astype(np.uint8), count

    # ----------------------------------------------------------
    # Tag 分析
    # ----------------------------------------------------------

    def _refine_homographies(self, homographies, tag_dets_by_cam, active_names, images=None):
        """全局优化：用多相机重叠 Tag 校正每台相机的 homography。
        
        使用近处 Tag（像素大、可靠）高权重，远处 Tag（像素小、噪声大）低权重。
        """
        if len(active_names) < 2:
            return homographies

        floor_tags = self._load_floor_tags()

        # 1. 收集重叠 Tag + 计算像素尺寸（对角线长度，越大越近越可靠）
        tag_cam_map = {}  # tid -> {cam: (u, v, pixel_size)}
        for name in active_names:
            for d in tag_dets_by_cam[name]:
                tid = d.tag_id
                if tid not in tag_cam_map:
                    tag_cam_map[tid] = {}
                # 像素尺寸 = Tag 在图像中的对角线长度
                corners = d.corners
                diag = np.linalg.norm(corners[0] - corners[2])  # 对角
                tag_cam_map[tid][name] = (d.center[0], d.center[1], diag)

        overlap_tags = {tid: cams for tid, cams in tag_cam_map.items() if len(cams) >= 2}
        if len(overlap_tags) < 4:
            print(f"    重叠 Tag 不足 ({len(overlap_tags)}), 跳过优化")
            return homographies

        # 2. 加权共识位置 — 像素大的 Tag 权重高
        tag_consensus = {}  # tid -> (consensus_x, consensus_y)
        tag_weights = {}    # tid -> total_weight
        for tid in overlap_tags:
            world_pts = []
            weights = []
            for name in overlap_tags[tid]:
                H = homographies.get(name)
                if H is None: continue
                try:
                    H_inv = np.linalg.inv(H)
                    du, dv, diag = tag_cam_map[tid][name]
                    wh = H_inv @ np.array([du, dv, 1.0])
                    wx, wy = wh[0]/wh[2], wh[1]/wh[2]
                    world_pts.append(np.array([wx, wy]))
                    # 权重 = 像素尺寸^2（面积），远处 Tag 像素小 → 权重低
                    weights.append(diag ** 2)
                except: pass
            if len(world_pts) >= 2:
                w = np.array(weights)
                w /= w.sum()
                tag_consensus[tid] = np.average(world_pts, axis=0, weights=w)
                tag_weights[tid] = sum(weights)

        if len(tag_consensus) < 4:
            return homographies

        # 3. 每台相机拟合加权仿射校正
        corrected = dict(homographies)
        for name in active_names:
            H = homographies.get(name)
            if H is None: continue

            src_pts = []; dst_pts = []; pt_weights = []
            for tid in overlap_tags:
                if name not in overlap_tags[tid] or tid not in tag_consensus:
                    continue
                cwx, cwy = tag_consensus[tid]
                du, dv, diag = tag_cam_map[tid][name]
                wh = H @ np.array([cwx, cwy, 1.0])
                cu, cv = wh[0]/wh[2], wh[1]/wh[2]
                src_pts.append(np.array([cu, cv]))
                dst_pts.append(np.array([du, dv]))
                # 权重: pixel_size^2 × consensus_weight
                pt_weights.append(diag ** 2 * tag_weights.get(tid, 1.0))

            if len(src_pts) < 3:
                continue

            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            pt_weights = np.array(pt_weights)

            # 加权仿射：按权重重复采样点
            w_sum = pt_weights.sum()
            if w_sum > 0:
                pt_weights_norm = pt_weights / w_sum
                # 按权重重复点（整数倍）
                repeats = np.maximum(1, (pt_weights_norm * 50).astype(int))
                src_w = np.repeat(src_pts, repeats, axis=0)
                dst_w = np.repeat(dst_pts, repeats, axis=0)

                A = cv2.estimateAffine2D(dst_w, src_w, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
                if A is None or A[0] is None:
                    A = cv2.estimateAffinePartial2D(dst_w, src_w, method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0)
            else:
                A = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
                if A is None or A[0] is None:
                    A = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0)

            if A is None or A[0] is None:
                continue

            A_mat = A[0]
            A3 = np.eye(3, dtype=np.float64)
            A3[:2, :] = A_mat.astype(np.float64)
            corrected[name] = A3 @ H
            scale = np.linalg.norm(A_mat[:, :2])
            shift = np.linalg.norm(A_mat[:, 2])
            print(f"    {self._name_to_label(name)}: scale={scale:.3f} shift={shift:.1f}px ({len(src_pts)} tags)")

        return corrected

    # ----------------------------------------------------------
    # Tag 分析
    # ----------------------------------------------------------

    def analyze_tags(self, camera_names, images, tag_dets_by_cam=None):
        """分析每个地面 Tag 在各相机的可见性和 GSD。
        
        如果提供 tag_dets_by_cam，按像素尺寸加权排序最优相机。
        返回 list[dict]：tag_id, x, y, visible_cameras, best_camera, gsd_by_cam, quality...
        """
        params = self.get_camera_params(camera_names)
        floor_tags = self._load_floor_tags()
        tag_data = []

        for tid in sorted(floor_tags.keys()):
            tx, ty, tz = floor_tags[tid]

            visible = {}
            pixel_sizes = {}  # 检测到的 Tag 像素尺寸，用于跨相机加权

            for name in camera_names:
                if name not in params or name not in images:
                    continue
                p = params[name]
                h_img, w_img = images[name].shape[:2]
                if _point_visible(tx, ty, tz, p["K"], p["R"], p["t"], w_img, h_img):
                    g = _gsd_at_point(tx, ty, tz, p["K"], p["R"], p["t"])
                    visible[name] = round(g, 2)

            # 如果有检测数据，用像素尺寸修正最优相机判断
            if tag_dets_by_cam:
                for name in camera_names:
                    if name in tag_dets_by_cam:
                        for d in tag_dets_by_cam[name]:
                            if d.tag_id == tid:
                                diag = np.linalg.norm(d.corners[0] - d.corners[2])
                                pixel_sizes[name] = diag
                                break

            if not visible:
                continue

            # 跨相机加权：像素大的相机 GSD 更可信
            if len(pixel_sizes) >= 2:
                # 按像素尺寸排序，最大的为最优
                best_name = max(pixel_sizes, key=pixel_sizes.get)
                best_gsd = visible.get(best_name, 0)
                # 质量分数 = 像素尺寸 / 最大尺寸 (0~1)
                max_size = max(pixel_sizes.values())
                quality = {n: round(s / max_size, 2) for n, s in pixel_sizes.items()}
            else:
                best_name = min(visible, key=visible.get)
                best_gsd = visible[best_name]
                quality = {n: 1.0 for n in visible}

            tag_data.append({
                "id": tid,
                "x": tx,
                "y": ty,
                "n_visible": len(visible),
                "best_camera": best_name,
                "gsd_by_cam": visible,
                "best_gsd": best_gsd,
                "pixel_sizes": {n: round(float(s), 1) for n, s in pixel_sizes.items()},
                "quality": quality,
            })

        return tag_data

    # ----------------------------------------------------------
    # 标注绘制
    # ----------------------------------------------------------

    def draw_annotations(self, fused, tag_data, camera_names, images=None):
        """在融合 BEV 图上绘制 Tag 标注、覆盖边框、相机位置、图例。
        直接修改 fused 图像。"""
        params = self.get_camera_params(camera_names)
        floor_tags = self._load_floor_tags()

        # --- Tag 标注 ---
        for ta in tag_data:
            u, v = _w2p(ta["x"], ta["y"],
                         self.x_min, self.x_max, self.y_min, self.y_max,
                         self.ppm, self.margin, self.out_h)
            if not (0 <= u < self.out_w and 0 <= v < self.out_h):
                continue

            # 颜色取最佳相机的颜色
            best = ta["best_camera"]
            dot_color = params.get(best, {}).get("color", (180, 180, 180))

            n_vis = ta["n_visible"]
            r = 4 + n_vis * 2  # 越多相机看到，圆点越大
            cv2.circle(fused, (u, v), r, dot_color, -1)
            cv2.circle(fused, (u, v), r + 1, (0, 0, 0), 1)

            if ta["id"] % 10 == 0:
                cv2.putText(fused, str(ta["id"]), (u + 6, v + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # --- 相机覆盖边框 ---
        for name in camera_names:
            if name not in params:
                continue
            # 需要重新生成 mask 来找轮廓
            # 但为了避免重复计算，如果 images 提供了，就生成 mask
            if images and name in images:
                img = images[name]
                p = params[name]
                _, mask = self._make_bev_undistorted(
                    img, p["K"], p["dist"], p["R"], p["t"])
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    color_bgr = params[name]["color"]
                    cv2.polylines(fused, [approx], True, color_bgr, 3)
                    # 标注相机名
                    x_min = approx[:, 0, 0].min()
                    y_min = approx[:, 0, 1].min()
                    label = self._name_to_label(name)
                    cv2.putText(fused, label, (x_min + 6, y_min + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

        # --- 相机位置 ---
        for name in camera_names:
            if name not in params:
                continue
            p = params[name]
            pos = (-p["R"].T @ p["t"]).flatten()
            h_cm = abs(pos[2]) * 100
            pu, pv = _w2p(pos[0], pos[1],
                           self.x_min, self.x_max, self.y_min, self.y_max,
                           self.ppm, self.margin, self.out_h)
            cv2.circle(fused, (pu, pv), 16, (0, 0, 0), 2)
            cv2.circle(fused, (pu, pv), 14, p["color"], -1)
            label = self._name_to_label(name)
            cv2.putText(fused, label, (pu + 18, pv - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, p["color"], 2)
            cv2.putText(fused, f"H={h_cm:.0f}cm", (pu + 18, pv + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(fused, f"({pos[0]:.2f},{pos[1]:.2f})", (pu + 18, pv + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1)

        # --- 图例 ---
        lx, ly = self.out_w - 240, 40
        cv2.rectangle(fused, (lx - 5, ly - 5), (lx + 235, ly + 30 + len(camera_names) * 22),
                      (30, 30, 30), -1)
        cv2.putText(fused, "TAG QUALITY (best GSD)", (lx, ly + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for i, name in enumerate(camera_names):
            if name not in params:
                continue
            cv2.circle(fused, (lx + 14, ly + 32 + i * 22), 5,
                       params[name]["color"], -1)
            label = self._name_to_label(name)
            cv2.putText(fused, label, (lx + 24, ly + 36 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # 若有多相机重叠，加注释
        if len(camera_names) > 1:
            y_leg = ly + 32 + len(camera_names) * 22
            cv2.circle(fused, (lx + 14, y_leg + 4), 4, (180, 180, 180), -1)
            cv2.putText(fused, f"{len(camera_names)}-cam overlap",
                        (lx + 24, y_leg + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # --- 比例尺 ---
        bar_y = self.out_h - 25
        cv2.line(fused, (self.margin, bar_y),
                 (self.margin + self.ppm, bar_y), (255, 255, 255), 4)
        cv2.putText(fused, "1m", (self.margin + 10, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 标题
        cam_list = ", ".join(self._name_to_label(n) for n in camera_names if n in params)
        title = f"{len(camera_names)}-Camera BEV  {self.ppm}px/m  {self.x_min}-{self.x_max}x{self.y_min}-{self.y_max}m"
        cv2.putText(fused, title, (self.margin, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(fused, f"[{cam_list}]", (self.margin, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def run(self, camera_names=None, images=None, file_map=None):
        """执行 BEV 生成 + Tag 分析。

        参数:
          camera_names: list[str] — 使用的相机名列表。None = 使用全部可用。
          images: dict[str, np.ndarray] — 相机图像。None 则从 file_map 或实时捕获。
          file_map: dict[str, str] — name -> file_path，用于离线加载。

        返回:
          (fused_image, tag_data, cam_stats, bevs, masks)
            fused_image: (H,W,3) BGR numpy array
            tag_data: list[dict] Tag 分析数据
            cam_stats: dict 各相机覆盖统计
            bevs: dict[name -> ndarray] 各相机单独的 BEV
            masks: dict[name -> ndarray] 各相机 cover mask
        """
        params = self._load_all_camera_params()

        if camera_names is None:
            camera_names = list(params.keys())
        else:
            # 过滤掉没有参数的相机
            camera_names = [n for n in camera_names if n in params]

        if not camera_names:
            raise ValueError("没有可用的相机")

        print(f"BEV 生成: {len(camera_names)} 个相机 -> {camera_names}")

        # 获取图像
        if images is None:
            if file_map is not None:
                images = self.load_images_from_files(file_map)
            else:
                images = self.capture_images(camera_names)

        # 过滤有图像的相机
        active_names = [n for n in camera_names if n in images]
        if not active_names:
            raise ValueError("没有可用的相机图像")

        print(f"活动相机: {active_names}")

        # --- 生成各相机 BEV ---
        bevs = {}
        masks = {}
        homographies = {}  # 收集每个相机的 homography
        tag_dets_by_cam = {}  # 收集每个相机的 Tag 检测用于全局优化
        # 尝试从地面 Tag 计算 homography（比标定更准）
        from pupil_apriltags import Detector
        try:
            detector = Detector(families="tag36h11", quad_decimate=1.0)
            clahe = cv2.createCLAHE(2.0, (8, 8))
        except:
            detector = None
            clahe = None

        for name in active_names:
            p = params[name]
            img = images[name]
            print(f"  投影 {name}...")

            # 先用 homography（如有足够地面 Tag）
            used_homography = False
            if detector is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scale = 0.5 if max(img.shape) > 2000 else 1.0
                gray_s = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
                gray_s = clahe.apply(gray_s)
                dets = detector.detect(gray_s)
                if scale != 1.0:
                    for d in dets:
                        d.corners /= scale
                        d.center = (d.center[0] / scale, d.center[1] / scale)

                floor_tags = self._load_floor_tags()
                fd = [d for d in dets if d.tag_id in floor_tags]
                tag_dets_by_cam[name] = fd  # 保存用于全局优化
                if len(fd) >= 4:
                    world_xy = np.array([floor_tags[d.tag_id][:2] for d in fd], dtype=np.float64)
                    img_uv = np.array([d.center for d in fd], dtype=np.float64)
                    H, _ = cv2.findHomography(world_xy, img_uv, cv2.RANSAC, 5.0)
                    if H is not None:
                        bev, mask = self._make_bev_from_homography(img, H)
                        homographies[name] = H
                        used_homography = True
                        if (mask > 0).sum() > 100:
                            print(f"    -> homography ({len(fd)} tags)")

            if not used_homography:
                bev, mask = self._make_bev_undistorted(img, p["K"], p["dist"], p["R"], p["t"])
                print(f"    -> extrinsics")

            bevs[name] = bev
            masks[name] = mask

        # --- 全局优化：用重叠 Tag 校正各相机 homography ---
        if len(active_names) >= 2 and homographies:
            print("  全局优化中...")
            homographies = self._refine_homographies(
                homographies, tag_dets_by_cam, active_names, images)
            # 用校正后的 homography 重新生成 BEV
            for name in active_names:
                if name in homographies:
                    p = params[name]
                    img = images[name]
                    bev, mask = self._make_bev_from_homography(img, homographies[name])
                    bevs[name] = bev
                    masks[name] = mask

        # --- 融合 ---
        print("  融合中...")
        fused, count = self._fuse_bevs(bevs, masks, homographies)
        if fused is None:
            raise RuntimeError("BEV 融合失败")

        # --- Tag 分析 ---
        print("  分析 Tag...")
        tag_data = self.analyze_tags(active_names, images, tag_dets_by_cam)

        # --- 标注 ---
        print("  标注中...")
        self.draw_annotations(fused, tag_data, active_names, images)

        # --- 统计 ---
        cam_stats = {}
        for name in active_names:
            pct = (masks[name] > 0).sum() / (self.out_w * self.out_h) * 100
            cam_stats[name] = {
                "coverage_pct": round(pct, 1),
                "color": params[name]["color"],
                "label": self._name_to_label(name),
            }

        n_cams_2 = sum(1 for t in tag_data if t["n_visible"] >= 2)
        n_cams_max = sum(1 for t in tag_data if t["n_visible"] == len(active_names))
        print(f"  Tag 总数: {len(tag_data)}  "
              f"≥2相机可见: {n_cams_2}  "
              f"{len(active_names)}相机重叠: {n_cams_max}")

        return fused, tag_data, cam_stats, bevs, masks

    # ----------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------

    def save_report(self, fused, tag_data, cam_stats, camera_names,
                    output_prefix="bev"):
        """保存 BEV 图像 + HTML 报告。"""
        params = self._load_all_camera_params()
        active_names = [n for n in camera_names if n in params]

        # 保存 BEV 图
        bev_path = f"{output_prefix}.jpg"
        cv2.imwrite(bev_path, fused)
        print(f"BEV 图像已保存: {bev_path}")

        # --- 统计 ---
        from collections import Counter
        best_counter = Counter(t["best_camera"] for t in tag_data)

        n_cams_max = len(active_names)
        n_overlap_all = sum(1 for t in tag_data if t["n_visible"] == n_cams_max)
        n_overlap_2 = sum(1 for t in tag_data if t["n_visible"] >= 2)

        # --- HTML 报告 ---
        rows = ""
        for t in tag_data:
            gsd_cells = ""
            for name in active_names:
                gsd_val = t["gsd_by_cam"].get(name, "-")
                gsd_cells += f'<td>{gsd_val}</td>\n'

            best_name = t["best_camera"]
            best_label = self._name_to_label(best_name)
            color_rgb = params[best_name]["color"]
            # BGR -> RGB for HTML
            color_hex = f"#{color_rgb[2]:02x}{color_rgb[1]:02x}{color_rgb[0]:02x}"

            rows += f'''<tr>
<td>{t["id"]}</td><td>{t["x"]:.2f}</td><td>{t["y"]:.2f}</td><td>{t["n_visible"]}</td>
{gsd_cells}
<td style="color:{color_hex};font-weight:bold">{best_label}</td>
</tr>\n'''

        # 表头 GSD 列
        gsd_headers = "".join(f'<th>{self._name_to_label(n)} GSD</th>' for n in active_names)

        # 相机颜色
        color_rows = "".join(
            f'<div class="card"><h3>{self._name_to_label(n)} Coverage</h3>'
            f'<div class="v" style="color:rgb({cam_stats[n]["color"][2]},{cam_stats[n]["color"][1]},{cam_stats[n]["color"][0]})">'
            f'{cam_stats[n]["coverage_pct"]}%</div></div>\n'
            for n in active_names if n in cam_stats
        )

        best_rows = "".join(
            f'<div class="card"><h3>{self._name_to_label(n)} Best</h3>'
            f'<div class="v" style="color:rgb({params[n]["color"][2]},{params[n]["color"][1]},{params[n]["color"][0]})">'
            f'{best_counter.get(n, 0)}</div></div>\n'
            for n in active_names if n in params
        )

        html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>{len(active_names)}-Camera BEV Tag Analysis</title>
<style>
body{{font-family:'Segoe UI',Arial,'Microsoft YaHei',sans-serif;margin:30px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#e94560;border-bottom:2px solid #e94560;padding-bottom:8px}}
h2{{background:#16213e;color:#e0e0e0;padding:8px 16px;border-left:3px solid #e94560}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#16213e;border:1px solid #2a2a4a;border-radius:8px;padding:14px 20px;min-width:120px}}
.card h3{{margin:0 0 6px;font-size:12px;color:#888;text-transform:uppercase}}
.card .v{{font-size:24px;font-weight:bold;color:#e0e0e0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th{{background:#0f3460;color:#e0e0e0;padding:8px 10px;text-align:left;position:sticky;top:0}}
td{{padding:5px 10px;border-bottom:1px solid #2a2a4a}}
tr:nth-child(even){{background:#1e1e35}}
img{{max-width:100%;border:1px solid #2a2a4a;border-radius:4px;margin:16px 0}}
.foot{{color:#666;font-size:11px;margin-top:30px;border-top:1px solid #2a2a4a;padding-top:16px}}
.explain{{background:#1a2a1a;border:1px solid #2a4a2a;border-radius:6px;padding:14px 20px;margin:16px 0;font-size:13px;line-height:1.8}}
.explain strong{{color:#55efc4}}
.explain ul{{margin:8px 0;padding-left:20px}}
.explain li{{margin:4px 0}}
</style></head><body>
<h1>{len(active_names)}-Camera BEV Tag Precision Analysis</h1>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
<strong>Cameras:</strong> {', '.join(self._name_to_label(n) for n in active_names)} &nbsp;|&nbsp;
<strong>Method:</strong> GSD (Ground Sampling Distance, mm/px) — smaller = better</p>

<div class="cards">
<div class="card"><h3>Tags in View</h3><div class="v">{len(tag_data)}</div></div>
<div class="card"><h3>{n_cams_max}-Cam Overlap</h3><div class="v">{n_overlap_all}</div></div>
<div class="card"><h3>≥2 Cam Overlap</h3><div class="v">{n_overlap_2}</div></div>
{color_rows}
{best_rows}
</div>

<h2>How to Read the BEV Image</h2>
<div class="explain">
<strong>俯视图含义：</strong>将各相机画面投影到地面，从正上方俯瞰拼接而成。<br>
<ul>
<li><strong>有颜色的区域</strong> = 至少一台相机能看到地面。颜色来自真实地板纹理。</li>
<li><strong>黑色区域</strong> = 没有相机覆盖的地面（相机视野外或被遮挡）。</li>
<li><strong>彩色圆点</strong> = 地面 AprilTag 位置。颜色代表 GSD 最优的相机：
  {"".join(f'<span style="color:rgb({params[n]["color"][2]},{params[n]["color"][1]},{params[n]["color"][0]})">● {self._name_to_label(n)}</span> ' for n in active_names if n in params)}
  圆点越大 = 越多相机能看到该 Tag。</li>
<li><strong>相机位置圆圈</strong> = 相机位置，标注了高度和世界坐标。</li>
</ul>
</div>

<h2>BEV Image</h2>
<img src="{bev_path}" alt="{len(active_names)}-Camera BEV">

<h2>Tag-by-Tag GSD (mm/px)</h2>
<p>GSD = 地面采样间距 (mm/px)，越小越好。颜色标注为该 Tag 位置 GSD 最优的相机。</p>
<table>
<tr><th>ID</th><th>X</th><th>Y</th><th>#Cams</th>{gsd_headers}<th>Best</th></tr>
{rows}
</table>
<div class="foot">ROS-Camera BEV Report — auto-generated by bev_generic.py</div>
</body></html>'''

        report_path = f"{output_prefix}_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML 报告已保存: {report_path}")

        return bev_path, report_path


# ==============================================================
# CLI 入口
# ==============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="通用 N 相机 BEV 俯视图融合")
    parser.add_argument("--cameras", type=str, default=None,
                        help="相机名列表，逗号分隔，如 picam_1,usb_cam_1  (默认: 全部)")
    parser.add_argument("--live", action="store_true",
                        help="实时捕获模式")
    parser.add_argument("--offline", type=str, default=None,
                        help="离线模式：name=path, 逗号分隔，如 picam_1=picam.jpg,usb_cam_1=usb1.jpg")
    parser.add_argument("--output", type=str, default="bev",
                        help="输出文件前缀 (默认: bev)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--extrinsics", type=str, default="extrinsics.yaml",
                        help="外参文件路径")
    parser.add_argument("--floor-tags", type=str, default="floor_tags.yaml",
                        help="地面 Tag 文件路径")

    args = parser.parse_args()

    gen = BevGenerator(
        config_path=args.config,
        extrinsics_path=args.extrinsics,
        floor_tags_path=args.floor_tags,
    )

    camera_names = None
    if args.cameras:
        camera_names = [n.strip() for n in args.cameras.split(",")]

    if args.offline:
        # 离线模式
        file_map = {}
        for pair in args.offline.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                file_map[name.strip()] = path.strip()
        fused, tag_data, cam_stats, _bevs, _masks = gen.run(
            camera_names=camera_names, file_map=file_map)
    else:
        # 实时模式
        fused, tag_data, cam_stats, _bevs, _masks = gen.run(
            camera_names=camera_names)

    gen.save_report(fused, tag_data, cam_stats, camera_names or gen.get_available_cameras(),
                    output_prefix=args.output)


if __name__ == "__main__":
    main()
