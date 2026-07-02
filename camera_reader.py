"""
统一相机读取接口 — 多 USB 相机读取
=====================================
统一封装 OpenCV VideoCapture，所有方法统一为 BGR 格式。"""

import platform
import numpy as np

_IS_WINDOWS = platform.system() == "Windows"

class CameraReader:
    """单相机读取封装。"""

    def __init__(self, cfg: dict):
        """
        cfg 字段（来自 config.yaml cameras[]）:
          name       — 相机标识名
          device     — 设备索引 "0", "1", "2" 等
          resolution — [width, height]
        """
        self.name = cfg["name"]
        self._device = cfg["device"]
        self._res = tuple(cfg.get("resolution", [640, 480]))
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
        # 先设 MJPG，再设分辨率（否则某些相机卡在 640×480）
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[1])
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
        """返回一帧 (H, W, 3) BGR ndarray，失败时返回 None。"""
        ret, frame = self._cam.read()
        return frame if ret else None

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
