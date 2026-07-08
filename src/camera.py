"""
统一相机读取接口 — 多 USB 相机读取
=====================================
统一封装 OpenCV VideoCapture，所有方法统一为 BGR 格式。
除了 MJPG 格式和分辨率，其他全用摄像头默认值。
"""

import platform
import subprocess as _subprocess
import os

_IS_WINDOWS = platform.system() == "Windows"


def _v4l2_preset(device_idx):
    """Linux V4L2 硬件参数预置：在 OpenCV 打开前用 v4l2-ctl 设置。
    OpenCV 的 CAP_PROP_AUTO_EXPOSURE/AUTO_WB 在 V4L2 上映射不正确
    （auto_exposure=1 实际是手动模式），必须用 v4l2-ctl 直接写硬件寄存器。
    """
    dev = f"/dev/video{device_idx}"
    if not os.path.exists(dev):
        return
    try:
        _subprocess.run(
            f"v4l2-ctl -d {dev} --set-ctrl="
            f"auto_exposure=3,"             # 光圈优先 = 自动曝光
            f"white_balance_automatic=1",    # 自动白平衡
            shell=True, capture_output=True, timeout=5)
    except Exception:
        pass  # v4l2-ctl 不可用时静默跳过


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
        # Linux: 先用 v4l2-ctl 预置硬件参数（OpenCV V4L2 映射不正确）
        if not _IS_WINDOWS:
            _v4l2_preset(idx)
        backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_V4L2
        cap = cv2.VideoCapture(idx, backend)
        # 只设 MJPG 格式和分辨率，其他用 v4l2-ctl 预置的参数
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
