#!/usr/bin/env python3
"""
多摄像头参数管理器

功能：
1. 统一管理多个摄像头的内参和外参
2. 支持多种摄像头类型：USB, PiCamera, ROS, 虚拟摄像头
3. 支持手动外参和自动标定两种模式
4. 提供统一的接口获取摄像头参数

使用方法：
    from camera_manager import CameraManager
    
    manager = CameraManager('cameras_config.yaml')
    manager.list_cameras()
    
    cam = manager.get_camera('usb_cam_1')
    k, dist = cam.get_intrinsics()
    r, t = cam.get_extrinsics()
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class Intrinsics:
    """摄像头内参"""
    camera_matrix: np.ndarray      # 3x3 相机矩阵
    distortion_coefficients: np.ndarray  # 畸变系数
    resolution: Tuple[int, int]   # 分辨率 (宽, 高)
    fps: int = 30                 # 帧率
    
    def validate(self) -> bool:
        """验证内参是否有效"""
        if self.camera_matrix.shape != (3, 3):
            return False
        if self.camera_matrix[2, 2] != 1.0:
            return False
        return True
    
    def get_fov(self) -> Tuple[float, float]:
        """获取水平/垂直视场角（度）"""
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        w, h = self.resolution
        
        # 计算每个像素的角度
        h_fov = 2 * np.arctan2(w / 2, fx)
        v_fov = 2 * np.arctan2(h / 2, fy)
        
        return np.degrees(h_fov), np.degrees(v_fov)


@dataclass
class Extrinsics:
    """摄像头外参"""
    mode: str  # "manual" | "auto_calibrate"
    r_world_camera: np.ndarray = field(default_factory=lambda: np.eye(3))  # 世界到相机旋转
    t_world_camera: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 世界到相机平移
    
    # 自动标定参数
    min_tags_required: int = 3
    confidence_threshold: float = 80.0
    update_rate: float = 1.0
    
    # 标定状态
    is_calibrated: bool = False
    last_calibration_time: float = 0.0
    calibration_confidence: float = 0.0
    
    def validate(self) -> bool:
        """验证外参是否有效"""
        if self.r_world_camera.shape != (3, 3):
            return False
        if self.t_world_camera.shape != (3,):
            return False
        # 检查旋转矩阵的正交性
        if not np.allclose(self.r_world_camera @ self.r_world_camera.T, np.eye(3), atol=1e-6):
            return False
        return True
    
    def set_from_euler(self, translation: List[float], rotation_euler: List[float]) -> None:
        """从欧拉角设置外参"""
        roll, pitch, yaw = rotation_euler
        
        # 依次应用 ZYX 顺序的旋转
        rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        self.r_world_camera = rz @ ry @ rx
        self.t_world_camera = np.array(translation, dtype=np.float64)
    
    def to_world_point(self, p_cam: np.ndarray) -> np.ndarray:
        """相机坐标系点转换到世界坐标系"""
        return self.r_world_camera.T @ (p_cam - self.t_world_camera)
    
    def to_camera_point(self, p_world: np.ndarray) -> np.ndarray:
        """世界坐标系点转换到相机坐标系"""
        return self.r_world_camera @ p_world + self.t_world_camera


@dataclass
class HardwareInfo:
    """硬件信息"""
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial: str = "N/A"
    mount_position: str = "unknown"  # ceiling | wall | floor | mobile


@dataclass
class Metadata:
    """元数据"""
    name: str = "Unnamed Camera"
    location: str = "Unknown"
    timezone: str = "UTC"
    notes: str = ""
    tags: List[str] = field(default_factory=list)


class CameraConfig:
    """单个摄像头配置"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.enabled = config.get('enabled', True)
        self.type = config.get('type', 'usb')
        
        # 加载内参
        intrinsics_cfg = config.get('intrinsics', {})
        k = np.array(intrinsics_cfg.get('camera_matrix', [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float64)
        dist = np.array(intrinsics_cfg.get('distortion_coefficients', [0,0,0,0,0]), dtype=np.float64).reshape(-1)
        res = tuple(config.get('resolution', [640, 480]))
        fps = int(config.get('fps', 30))
        
        self.intrinsics = Intrinsics(k, dist, res, fps)
        
        # 加载外参
        extrinsics_cfg = config.get('extrinsics', {})
        mode = extrinsics_cfg.get('mode', 'manual')
        
        self.extrinsics = Extrinsics(mode=mode)
        
        # 如果是手动模式，设置初始外参
        if mode == 'manual':
            t_cfg = extrinsics_cfg.get('T_world_camera', {})
            translation = t_cfg.get('translation', [0, 0, 0])
            rotation = t_cfg.get('rotation_euler', [0, 0, 0])
            self.extrinsics.set_from_euler(translation, rotation)
            self.extrinsics.is_calibrated = True
        else:
            # 自动标定模式
            auto_cfg = extrinsics_cfg.get('auto_calibration', {})
            self.extrinsics.min_tags_required = auto_cfg.get('min_tags_required', 3)
            self.extrinsics.confidence_threshold = auto_cfg.get('confidence_threshold', 80.0)
            self.extrinsics.update_rate = auto_cfg.get('update_rate', 1.0)
        
        # 硬件信息
        hw_cfg = config.get('hardware', {})
        self.hardware = HardwareInfo(
            manufacturer=hw_cfg.get('manufacturer', 'Unknown'),
            model=hw_cfg.get('model', 'Unknown'),
            serial=hw_cfg.get('serial', 'N/A'),
            mount_position=hw_cfg.get('mount_position', 'unknown')
        )
        
        # 元数据
        meta_cfg = config.get('metadata', {})
        self.metadata = Metadata(
            name=meta_cfg.get('name', name),
            location=meta_cfg.get('location', 'Unknown'),
            timezone=meta_cfg.get('timezone', 'UTC'),
            notes=meta_cfg.get('notes', ''),
            tags=meta_cfg.get('tags', [])
        )
        
        # 类型特定配置
        self._type_config = config
    
    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取内参矩阵和畸变系数"""
        return self.intrinsics.camera_matrix, self.intrinsics.distortion_coefficients
    
    def get_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取外参旋转矩阵和平移向量"""
        return self.extrinsics.r_world_camera, self.extrinsics.t_world_camera
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        return self.intrinsics.resolution
    
    def get_fps(self) -> int:
        """获取帧率"""
        return self.intrinsics.fps
    
    def is_static(self) -> bool:
        """判断是否为静态摄像头（不需要自动标定）"""
        return self.extrinsics.mode == 'manual'
    
    def needs_auto_calibration(self) -> bool:
        """是否需要自动标定"""
        return self.extrinsics.mode == 'auto_calibrate'
    
    def __repr__(self) -> str:
        status = "OK" if self.enabled else "DISABLED"
        calibration = "calibrated" if self.extrinsics.is_calibrated else "uncalibrated"
        return f"Camera({self.name}, type={self.type}, status={status}, {calibration})"


# =============================================================================
# 摄像头管理器
# =============================================================================

class CameraManager:
    """
    多摄像头管理器
    
    功能：
    - 加载和管理多个摄像头配置
    - 支持不同类型的摄像头
    - 管理内参和外参
    - 支持自动标定
    """
    
    def __init__(self, config_path: str = "cameras_config.yaml"):
        self.config_path = Path(config_path)
        self.cameras: Dict[str, CameraConfig] = {}
        self.floor_tag_world_centers: Dict[int, np.ndarray] = {}
        self._detector = None
        self._floor_tag_config_path = "floor_tag_layout.yaml"
        
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载全局设置
        global_cfg = config.get('global', {})
        self._floor_tag_config_path = global_cfg.get('floor_tag_layout', 'floor_tag_layout.yaml')
        
        # 加载摄像头配置
        cameras_cfg = config.get('cameras', {})
        for name, cam_cfg in cameras_cfg.items():
            self.cameras[name] = CameraConfig(name, cam_cfg)
        
        # 加载地面标签配置
        self._load_floor_tags()
        
        print(f"[CameraManager] Loaded {len(self.cameras)} cameras")
    
    def _load_floor_tags(self) -> None:
        """加载地面标签布局"""
        floor_tag_path = Path(self._floor_tag_config_path)
        if not floor_tag_path.exists():
            print(f"[CameraManager] Warning: Floor tag layout not found: {floor_tag_path}")
            return
        
        with open(floor_tag_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        for key, value in cfg.items():
            tag_id = int(key)
            if isinstance(value, list) and len(value) == 3:
                self.floor_tag_world_centers[tag_id] = np.array(value, dtype=np.float64)
        
        print(f"[CameraManager] Loaded {len(self.floor_tag_world_centers)} floor tags")
    
    def list_cameras(self) -> List[str]:
        """列出所有摄像头名称"""
        return list(self.cameras.keys())
    
    def list_enabled_cameras(self) -> List[str]:
        """列出所有启用的摄像头"""
        return [name for name, cam in self.cameras.items() if cam.enabled]
    
    def get_camera(self, name: str) -> Optional[CameraConfig]:
        """获取指定摄像头"""
        return self.cameras.get(name)
    
    def get_camera_intrinsics(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取摄像头内参"""
        cam = self.cameras.get(name)
        if cam is None:
            return None
        return cam.get_intrinsics()
    
    def get_camera_extrinsics(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取摄像头外参"""
        cam = self.cameras.get(name)
        if cam is None:
            return None
        return cam.get_extrinsics()
    
    def set_camera_extrinsics(self, name: str, r: np.ndarray, t: np.ndarray) -> bool:
        """设置摄像头外参"""
        cam = self.cameras.get(name)
        if cam is None:
            return False
        
        cam.extrinsics.r_world_camera = r.copy()
        cam.extrinsics.t_world_camera = t.copy()
        cam.extrinsics.is_calibrated = True
        cam.extrinsics.last_calibration_time = time.time()
        return True
    
    def update_camera_extrinsics_from_tags(
        self, 
        name: str, 
        detections: List[Any],
        min_tags: int = 3
    ) -> bool:
        """
        从AprilTag检测更新摄像头外参
        
        Args:
            name: 摄像头名称
            detections: AprilTag检测列表
            min_tags: 最少需要的标签数
        
        Returns:
            是否成功更新
        """
        cam = self.cameras.get(name)
        if cam is None:
            return False
        
        # 收集标签对应点
        world_pts = []
        cam_pts = []
        
        for det in detections:
            tag_id = getattr(det, 'tag_id', None)
            if tag_id is None:
                continue
            if tag_id not in self.floor_tag_world_centers:
                continue
            
            world_pts.append(self.floor_tag_world_centers[tag_id])
            tvec = getattr(det, 'tvec', None)
            if tvec is not None:
                cam_pts.append(np.array(tvec).reshape(3))
        
        if len(world_pts) < min_tags:
            return False
        
        # 估计变换
        r_wc, t_wc = self._estimate_world_to_camera_transform(
            np.vstack(world_pts),
            np.vstack(cam_pts)
        )
        
        cam.extrinsics.r_world_camera = r_wc
        cam.extrinsics.t_world_camera = t_wc
        cam.extrinsics.is_calibrated = True
        cam.extrinsics.last_calibration_time = time.time()
        cam.extrinsics.calibration_confidence = len(world_pts) / min_tags * 100
        
        return True
    
    def _estimate_world_to_camera_transform(
        self, 
        world_points: np.ndarray, 
        camera_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """估计世界到相机的刚体变换"""
        if world_points.shape[0] < 3:
            raise ValueError("At least 3 point pairs are required")
        
        centroid_a = np.mean(world_points, axis=0)
        centroid_b = np.mean(camera_points, axis=0)
        
        a_centered = world_points - centroid_a
        b_centered = camera_points - centroid_b
        
        h = a_centered.T @ b_centered
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        
        t = centroid_b - r @ centroid_a
        return r, t
    
    def get_calibration_status(self, name: str) -> Dict[str, Any]:
        """获取摄像头标定状态"""
        cam = self.cameras.get(name)
        if cam is None:
            return {"error": "Camera not found"}
        
        return {
            "name": name,
            "type": cam.type,
            "enabled": cam.enabled,
            "mode": cam.extrinsics.mode,
            "is_calibrated": cam.extrinsics.is_calibrated,
            "last_calibration": cam.extrinsics.last_calibration_time,
            "confidence": cam.extrinsics.calibration_confidence,
            "translation": cam.extrinsics.t_world_camera.tolist(),
            "rotation_euler": self._rotation_to_euler(cam.extrinsics.r_world_camera)
        }
    
    def _rotation_to_euler(self, r: np.ndarray) -> List[float]:
        """旋转矩阵转欧拉角 (roll, pitch, yaw)"""
        # 从旋转矩阵提取欧拉角
        sy = np.sqrt(r[0,0]**2 + r[1,0]**2)
        
        if sy > 1e-6:
            roll = np.arctan2(r[2,1], r[2,2])
            pitch = np.arctan2(-r[2,0], sy)
            yaw = np.arctan2(r[1,0], r[0,0])
        else:
            roll = np.arctan2(-r[1,2], r[1,1])
            pitch = np.arctan2(-r[2,0], sy)
            yaw = 0
        
        return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]
    
    def export_calibration_results(self, output_path: str) -> None:
        """导出标定结果到YAML文件"""
        results = {"cameras": {}}
        
        for name, cam in self.cameras.items():
            r, t = cam.get_extrinsics()
            euler = self._rotation_to_euler(r)
            
            results["cameras"][name] = {
                "intrinsics": {
                    "camera_matrix": cam.intrinsics.camera_matrix.tolist(),
                    "distortion_coefficients": cam.intrinsics.distortion_coefficients.tolist(),
                    "resolution": list(cam.intrinsics.resolution),
                    "fps": cam.intrinsics.fps
                },
                "extrinsics": {
                    "translation": t.tolist(),
                    "rotation_euler_deg": euler,
                    "is_calibrated": cam.extrinsics.is_calibrated,
                    "calibration_time": cam.extrinsics.last_calibration_time,
                    "confidence": cam.extrinsics.calibration_confidence
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[CameraManager] Exported calibration results to {output_path}")
    
    def print_status(self) -> None:
        """打印所有摄像头状态"""
        print("\n" + "="*70)
        print("摄像头管理器状态")
        print("="*70)
        
        print(f"\n总共: {len(self.cameras)} 个摄像头, {len(self.list_enabled_cameras())} 个启用")
        print(f"地面标签数: {len(self.floor_tag_world_centers)}")
        
        for name, cam in self.cameras.items():
            print(f"\n{cam}")
            print(f"  分辨率: {cam.intrinsics.resolution}, FPS: {cam.intrinsics.fps}")
            h_fov, v_fov = cam.intrinsics.get_fov()
            print(f"  视场角: {h_fov:.1f}° x {v_fov:.1f}°")
            print(f"  内参矩阵 fx={cam.intrinsics.camera_matrix[0,0]:.1f}, fy={cam.intrinsics.camera_matrix[1,1]:.1f}")
            print(f"  位置: {cam.extrinsics.t_world_camera}")
            print(f"  挂载: {cam.hardware.mount_position}")
            print(f"  元数据: {cam.metadata.name} @ {cam.metadata.location}")
        
        print("\n" + "="*70)


# =============================================================================
# 摄像头捕获器（针对不同类型）
# =============================================================================

class CameraCapture:
    """摄像头捕获器基类"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture = None
    
    def open(self) -> bool:
        """打开摄像头"""
        raise NotImplementedError
    
    def read(self) -> Optional[np.ndarray]:
        """读取帧"""
        raise NotImplementedError
    
    def close(self) -> None:
        """关闭摄像头"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def is_opened(self) -> bool:
        """检查是否打开"""
        return self._capture is not None and self._capture.isOpened()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class USBCameraCapture(CameraCapture):
    """USB摄像头捕获器"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
    
    def open(self) -> bool:
        if cv2 is None:
            raise ImportError("OpenCV not installed")
        
        device_index = self.config._type_config.get('device_index', 0)
        self._capture = cv2.VideoCapture(device_index)
        
        if not self._capture.isOpened():
            # 尝试 DSHOW 后端
            self._capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        
        if not self._capture.isOpened():
            return False
        
        # 设置分辨率和帧率
        w, h = self.config.intrinsics.resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        self._capture.set(cv2.CAP_PROP_FPS, float(self.config.intrinsics.fps))
        
        # 设置曝光（如果配置了）
        if not self.config._type_config.get('auto_exposure', True):
            exposure = self.config._type_config.get('exposure', -5)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        
        ok, frame = self._capture.read()
        return frame if ok else None


class PiCameraCapture(CameraCapture):
    """树莓派PiCamera捕获器（支持Picamera2）"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._picam2 = None
        self._use_picamera2 = False
    
    def _get_video_devices(self) -> List[str]:
        """获取当前 PiCamera 可能占用的视频设备"""
        configured = self.config._type_config.get('video_devices')
        if isinstance(configured, list) and configured:
            return [str(dev) for dev in configured]
        return ['/dev/video2', '/dev/video3']
    
    def _find_device_user_pids(self, device_path: str) -> List[int]:
        """查找占用设备的进程 PID"""
        if not Path(device_path).exists():
            return []
        
        try:
            result = subprocess.run(
                ['fuser', device_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        
        output = f"{result.stdout} {result.stderr}".strip()
        pids: List[int] = []
        for token in output.replace(':', ' ').split():
            if token.isdigit():
                pid = int(token)
                if pid != os.getpid():
                    pids.append(pid)
        return sorted(set(pids))
    
    def _release_busy_devices(self) -> None:
        """尝试释放被其他进程占用的 PiCamera 设备"""
        if not self.config._type_config.get('auto_release_device_users', True):
            return
        
        all_pids = set()
        for device_path in self._get_video_devices():
            for pid in self._find_device_user_pids(device_path):
                all_pids.add(pid)
        
        if not all_pids:
            return
        
        print(f"[{self.config.name}] Releasing busy camera devices, pids={sorted(all_pids)}")
        
        for pid in sorted(all_pids):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                print(f"[{self.config.name}] No permission to stop pid {pid}")
        
        time.sleep(1.0)
        
        for pid in sorted(all_pids):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
            
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                print(f"[{self.config.name}] No permission to kill pid {pid}")
        
        time.sleep(0.5)
    
    def open(self) -> bool:
        self._release_busy_devices()
        
        # 优先尝试使用 Picamera2（新版，官方推荐）
        try:
            from picamera2 import Picamera2
            self._use_picamera2 = True
            return self._open_picamera2()
        except ImportError:
            pass
        
        # 回退到旧版 picamera
        try:
            import picamera
            self._use_picamera2 = False
            return self._open_picamera_legacy()
        except ImportError:
            raise ImportError(
                "Neither picamera2 nor picamera is installed. "
                "Install picamera2: sudo apt install -y python3-picamera2"
            )
    
    def _open_picamera2(self) -> bool:
        """使用 Picamera2 打开相机（对齐 xxx.py 的稳定配置）"""
        from picamera2 import Picamera2
        
        self._picam2 = Picamera2()
        
        # 获取配置参数
        res = self.config._type_config.get('resolution', [640, 480])
        fps = self.config._type_config.get('framerate', 30)
        
        frame_us = int(round(1_000_000 / max(fps, 1)))
        
        base_controls = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": min(10000, frame_us),
            "AnalogueGain": 1.0,
        }
        enhanced_controls = {
            **base_controls,
            "NoiseReductionMode": 0,
            "Sharpness": 0.0,
        }
        
        try:
            config = self._picam2.create_video_configuration(
                main={"size": tuple(res), "format": "YUV420"},
                buffer_count=1,
                controls=enhanced_controls,
            )
        except Exception:
            config = self._picam2.create_video_configuration(
                main={"size": tuple(res), "format": "YUV420"},
                buffer_count=1,
                controls=base_controls,
            )
        
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(0.5)
        
        return True
    
    def _open_picamera_legacy(self) -> bool:
        """使用旧版 picamera 打开相机"""
        import picamera
        import picamera.array
        
        self._picamera = picamera.PiCamera()
        
        # 设置分辨率
        res = self.config._type_config.get('resolution', [640, 480])
        self._picamera.resolution = tuple(res)
        
        # 设置帧率
        fps = self.config._type_config.get('framerate', 30)
        self._picamera.framerate = fps
        
        # 设置白平衡和曝光
        awb = self.config._type_config.get('awb_mode', 'auto')
        exposure = self.config._type_config.get('exposure_mode', 'auto')
        
        if awb != 'auto':
            self._picamera.awb_mode = awb
        if exposure != 'auto':
            self._picamera.exposure_mode = exposure
        
        # 创建流式捕获
        self._stream = picamera.array.PiRGBArray(self._picamera, size=tuple(res))
        self._iterator = self._picamera.capture_continuous(
            self._stream, 
            format='bgr',
            use_video_port=True
        )
        
        # 预热
        import time
        time.sleep(2)
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        if self._use_picamera2:
            return self._read_picamera2()
        else:
            return self._read_picamera_legacy()
    
    def _read_picamera2(self) -> Optional[np.ndarray]:
        """从 Picamera2 读取帧（对齐 xxx.py 的 YUV420 读取方式）"""
        if self._picam2 is None:
            return None
        
        try:
            req = self._picam2.capture_request()
            try:
                frame_raw = req.make_array("main")
            finally:
                req.release()
            
            if frame_raw is None:
                return None
            
            res = self.config._type_config.get('resolution', [640, 480])
            width, height = int(res[0]), int(res[1])
            
            if frame_raw.ndim == 2:
                gray = frame_raw[:height, :width]
            elif frame_raw.ndim == 3:
                gray = frame_raw[:, :, 0]
            else:
                return None
            
            if cv2 is not None:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return np.repeat(gray[:, :, None], 3, axis=2)
        except Exception:
            return None
    
    def _read_picamera_legacy(self) -> Optional[np.ndarray]:
        """从旧版 picamera 读取帧"""
        if not hasattr(self, '_iterator') or self._iterator is None:
            return None
        
        try:
            next(self._iterator)
            frame = self._stream.array
            self._stream.truncate(0)
            return frame.copy()
        except Exception:
            return None
    
    def close(self) -> None:
        """关闭摄像头"""
        if self._use_picamera2:
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except:
                    pass
                self._picam2 = None
        else:
            if hasattr(self, '_picamera') and self._picamera is not None:
                try:
                    self._picamera.close()
                except:
                    pass
                self._picamera = None
            self._stream = None
            self._iterator = None


class ROSCameraCapture(CameraCapture):
    """ROS摄像头捕获器"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._node = None
        self._subscriber = None
        self._latest_frame = None
        self._lock = None
    
    def open(self) -> bool:
        try:
            import rclpy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
        except ImportError:
            raise ImportError("ROS2 not installed. Install: pip install rclpy sensor_msgs cv_bridge")
        
        topic = self.config._type_config.get('topic', '/camera/image_raw')
        frame_id = self.config._type_config.get('frame_id', 'camera_link')
        
        # 初始化 ROS 节点
        rclpy.init(args=None)
        self._node = rclpy.create_node('camera_capture_' + self.config.name)
        
        self._bridge = CvBridge()
        
        def callback(msg: Image):
            try:
                self._latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                pass
        
        self._subscriber = self._node.create_subscription(
            Image, topic, callback, qos_profile=10
        )
        
        # 创建异步处理线程
        import threading
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._spin_thread)
        self._thread.start()
        
        return True
    
    def _spin_thread(self):
        import rclpy
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
    
    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame
    
    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        if self._node:
            self._node.destroy_node()
        try:
            import rclpy
            rclpy.shutdown()
        except:
            pass


class VirtualCameraCapture(CameraCapture):
    """虚拟摄像头（视频文件）捕获器"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
    
    def open(self) -> bool:
        if cv2 is None:
            raise ImportError("OpenCV not installed")
        
        video_source = self.config._type_config.get('video_source', '')
        self._capture = cv2.VideoCapture(video_source)
        
        if not self._capture.isOpened():
            return False
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        
        ok, frame = self._capture.read()
        
        # 循环播放
        if not ok and self.config._type_config.get('loop', True):
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        
        return frame if ok else None


def create_camera_capture(config: CameraConfig) -> CameraCapture:
    """工厂函数：创建对应类型的摄像头捕获器"""
    capture_map = {
        'usb': USBCameraCapture,
        'picamera': PiCameraCapture,
        'ros': ROSCameraCapture,
        'virtual': VirtualCameraCapture
    }
    
    capture_class = capture_map.get(config.type)
    if capture_class is None:
        raise ValueError(f"Unknown camera type: {config.type}")
    
    return capture_class(config)


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """测试摄像头管理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Manager")
    parser.add_argument('--config', default='cameras_config.yaml', help='Config file path')
    parser.add_argument('--export', help='Export calibration results to file')
    parser.add_argument('--status', action='store_true', help='Print camera status')
    parser.add_argument('--camera', help='Show specific camera calibration status')
    
    args = parser.parse_args()
    
    try:
        manager = CameraManager(args.config)
        
        if args.status:
            manager.print_status()
        elif args.camera:
            status = manager.get_calibration_status(args.camera)
            print(f"\n{args.camera} 状态:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        elif args.export:
            manager.export_calibration_results(args.export)
        else:
            manager.print_status()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
