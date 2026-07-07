"""
统一相机读取接口 — 多 USB 相机读取
=====================================
统一封装 OpenCV VideoCapture，所有方法统一为 BGR 格式。
支持亮度/对比度/饱和度/增益控制，可选色彩增强（去灰蒙去黄）。
"""

import platform
import numpy as np

_IS_WINDOWS = platform.system() == "Windows"


def enhance_color(frame):
    """色彩增强：去灰蒙、去黄、提饱和度。
    
    - LAB L 通道 CLAHE → 提对比度
    - 灰度世界白平衡 → 去色偏
    - HSV S 通道拉伸 → 加饱和度
    """
    import cv2

    # 1. 白平衡：灰度世界假设（各通道均值拉平）
    b_mean, g_mean, r_mean = frame.mean(axis=(0, 1))
    gray = (b_mean + g_mean + r_mean) / 3.0
    frame_f = frame.astype(np.float32)
    frame_f[:, :, 0] *= np.clip(gray / max(b_mean, 1), 0.6, 1.6)
    frame_f[:, :, 1] *= np.clip(gray / max(g_mean, 1), 0.6, 1.6)
    frame_f[:, :, 2] *= np.clip(gray / max(r_mean, 1), 0.6, 1.6)
    frame = frame_f.clip(0, 255).astype(np.uint8)

    # 2. LAB L 通道 CLAHE → 提对比度，去灰蒙
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. HSV S 通道拉伸 → 加饱和度
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return frame


class CameraReader:
    """单相机读取封装。"""

    def __init__(self, cfg: dict):
        """
        cfg 字段（来自 config.yaml cameras[]）:
          name       — 相机标识名
          device     — 设备索引 "0", "1", "2" 等
          resolution — [width, height]
          brightness — 亮度 (默认 0)
          contrast   — 对比度 (默认 32)
          saturation — 饱和度 (默认 64)
          gain       — 增益 (默认 16)
          enhance    — 是否色彩增强 (默认 false)
        """
        self.name = cfg["name"]
        self._device = cfg["device"]
        self._res = tuple(cfg.get("resolution", [640, 480]))
        self._brightness = cfg.get("brightness", 0)
        self._contrast = cfg.get("contrast", 32)
        self._saturation = cfg.get("saturation", 64)
        self._gain = cfg.get("gain", 16)
        self._enhance = cfg.get("enhance", False)
        self._cam = None

    # ------------------------------------------------------------------
    def open(self):
        """打开相机。返回 self 方便链式调用。"""
        return self._open_usb()

    def _open_usb(self):
        import cv2
        idx = int(self._device) if self._device.isdigit() else self._device
        backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_V4L2
        cap = cv2.VideoCapture(idx, backend)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[1])
        # 设置曝光/颜色参数
        if self._brightness != 0:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self._brightness)
        if self._contrast != 32:
            cap.set(cv2.CAP_PROP_CONTRAST, self._contrast)
        if self._saturation != 64:
            cap.set(cv2.CAP_PROP_SATURATION, self._saturation)
        if self._gain != 0:
            cap.set(cv2.CAP_PROP_GAIN, self._gain)
        # 确认实际打开的分辨率
        real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"[{self.name}] 无法打开 USB 相机 {self._device}")
        print(f"  [{self.name}] USB {self._device} -> {real_w:.0f}x{real_h:.0f}"
              f" {'DSHOW' if _IS_WINDOWS else 'V4L2'}")
        self._cam = cap

    # ------------------------------------------------------------------
    def read(self):
        """返回一帧 (H, W, 3) BGR ndarray，失败时返回 None。
        如果 enhance=True，自动做色彩增强（白平衡 + 去灰蒙 + 提饱和度）。
        """
        ret, frame = self._cam.read()
        if not ret:
            return None
        if self._enhance:
            frame = enhance_color(frame)
        return frame

    # ------------------------------------------------------------------
    def release(self):
        if self._cam is None:
            return
        self._cam.release()
        self._cam = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


# ======================================================================
def open_all_cameras(config: dict) -> list[CameraReader]:
    """根据 config['cameras'] 打开所有相机，返回 CameraReader 列表。"""
    import time
    readers = []
    for cam_cfg in config["cameras"]:
        try:
            r = CameraReader(cam_cfg).open()
            readers.append(r)
            print(f"[相机] {r.name} 已打开")
        except Exception as e:
            print(f"[相机] {cam_cfg['name']} 打开失败: {e}")
    return readers


def close_all_cameras(readers: list[CameraReader]):
    """关闭所有相机。"""
    for r in readers:
        try:
            r.release()
            print(f"[相机] {r.name} 已关闭")
        except Exception as e:
            print(f"[相机] {r.name} 关闭异常: {e}")

