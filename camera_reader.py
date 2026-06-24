"""
统一相机读取接口 — ROS-Camera 多相机立方体追踪
================================================
支持 USB 相机 (OpenCV VideoCapture) 和 Raspberry Pi Camera (Picamera2)。
所有方法统一为 image array (H, W, 3) BGR 格式。"""

import platform
import numpy as np

_IS_WINDOWS = platform.system() == "Windows"

class CameraReader:
    """单相机读取封装。"""

    def __init__(self, cfg: dict):
        """
        cfg 字段（来自 config.yaml cameras[]）:
          name       — 相机标识名
          type       — "usb" | "picamera"
          device     — USB: 设备索引 "0" 或 "/dev/video0"; PiCamera: "picamera:0"
          resolution — [width, height]
        """
        self.name = cfg["name"]
        self._type = cfg["type"]
        self._device = cfg["device"]
        self._res = tuple(cfg.get("resolution", [640, 480]))
        self._cam = None

    # ------------------------------------------------------------------
    def open(self):
        """打开相机。返回 self 方便链式调用。"""
        if self._type == "picamera":
            self._open_picamera()
        else:
            self._open_usb()
        return self

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

    def _open_picamera(self):
        from picamera2 import Picamera2
        idx = int(self._device.split(":")[1])
        picam = Picamera2(idx)
        cfg = picam.create_video_configuration(main={"size": self._res})
        picam.configure(cfg)
        picam.start()
        self._cam = picam

    # ------------------------------------------------------------------
    def read(self):
        """返回一帧 (H, W, 3) BGR ndarray，失败时返回 None。"""
        if self._type == "picamera":
            return self._cam.capture_array()       # RGB888 配置下实际输出 BGR
        ret, frame = self._cam.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    def release(self):
        if self._cam is None:
            return
        if self._type == "picamera":
            self._cam.stop()
        else:
            self._cam.release()
        self._cam = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


# ======================================================================
def open_all_cameras(config: dict) -> list[CameraReader]:
    """根据 config['cameras'] 打开所有相机，返回 CameraReader 列表。

    **重要**：PiCamera 必须最先初始化，否则会与 USB 相机的驱动冲突。
    """
    import time
    # 排序：picamera 优先
    cameras_sorted = sorted(config["cameras"],
                           key=lambda c: 0 if c["type"] == "picamera" else 1)
    readers = []
    for cam_cfg in cameras_sorted:
        try:
            r = CameraReader(cam_cfg).open()
            readers.append(r)
            print(f"[相机] {r.name} 已打开 ({cam_cfg['type']})")
            if cam_cfg["type"] == "picamera":
                time.sleep(0.5)  # 等 PiCamera 稳定后再开下一个
        except Exception as e:
            print(f"[相机] {cam_cfg['name']} 打开失败: {e}")
    return readers


def close_all_cameras(readers: list[CameraReader]):
    """关闭所有相机，USB 先关、PiCamera 最后关（与打开顺序相反）。"""
    # 排序：picamera 最后关
    for r in sorted(readers, key=lambda c: 0 if c._type == "picamera" else 1, reverse=True):
        try:
            r.release()
            print(f"[相机] {r.name} 已关闭")
        except Exception as e:
            print(f"[相机] {r.name} 关闭异常: {e}")
