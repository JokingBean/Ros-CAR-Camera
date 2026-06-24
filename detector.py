"""
AprilTag 检测封装 — ROS-Camera 多相机立方体追踪
================================================
基于 pupil-apriltags 库，支持自定义 Tag 族、检测结果过滤。"""

import cv2
from pupil_apriltags import Detector

# ------------------------------------------------------------------
class TagDetector:
    """AprilTag 检测器（线程安全，无状态）。"""

    def __init__(self, families: str = "tag36h11"):
        self._detector = Detector(families=families,
                                 quad_decimate=1.5,
                                 decode_sharpening=0.5)

    def detect(self, frame):
        """检测一帧图像中的 AprilTag。

        参数:
          frame — (H, W, 3) BGR ndarray

        返回 pupil_apriltags.Detection 列表，每个含 .tag_id / .corners / .center。
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return self._detector.detect(gray)
