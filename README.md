# ROS-Camera 多相机立方体追踪系统

## 系统架构说明

### 核心特性
- **静态相机部署**：所有相机固定安装，无动态移动
- **全局外参标定**：启动时自动校验，误差超标自动重新标定
- **多相机时序同步**：Picamera2硬件时间戳 + USB软件窗口匹配（±25ms）
- **全局BA优化**：基于multi-cam-apriltag-calib的光束平差算法
- **持久化外参**：标定结果保存至global_extrinsics.yaml，长期复用

## 启动流程

```
程序启动
├─ 加载 config/global_extrinsics.yaml（历史外参）
├─ 多线程同步采集10组地面Tag帧
├─ 计算全体相机平均重投影误差
│  ├─ 误差 ≤ 1.2px：校验通过 → 直接进入目标跟踪
│  └─ 误差 > 1.2px：自动重新标定
│     ├─ 采集30组同步Tag图集
│     ├─ 执行全局BA优化
│     ├─ 覆盖保存 global_extrinsics.yaml
│     └─ 加载新外参进入跟踪
└─ 运行阶段：固定外参，不再更新
```

## 目录结构

```
ROS-Camera/
├── camera_io/
│   └── camera_system.py          # 多线程相机采集+时序同步
├── calibration/
│   ├── calib_check.py            # 开机外参校验
│   └── auto_ba_calib.py          # 全局BA标定封装
├── detection/
│   └── apriltag_detector.py      # AprilTag检测（局部去畸变）
├── fusion_tracking/
│   └── multi_camera_fusion.py    # 多观测融合+卡尔曼滤波
├── visualization/
│   └── stitch_view.py            # BEV鸟瞰图（可选）
├── config/
│   ├── cameras_config.yaml       # 相机内参+系统配置
│   └── global_extrinsics.yaml    # 全局外参（自动生成）
├── logs/                         # 运行日志
├── tracker_main.py               # 主程序入口
└── requirements.txt
```

## 配置文件

### cameras_config.yaml 新增配置

```yaml
# 时序同步配置
timing:
  max_frame_diff_ms: 25           # 多相机时间戳最大容差
  frame_queue_size: 30            # 帧缓存队列大小

# 外参校验配置
calibration_check:
  max_allowed_repro_error: 1.2    # 重投影误差阈值（像素）
  check_frame_num: 10             # 启动校验帧数
  auto_capture_count: 30          # 自动重标定采集帧数
  min_floor_tags: 8               # 最少地面Tag数量

# 运行模式
runtime:
  enable_bev_view: true           # 是否启用BEV俯视图
  target_fps: 20                  # 目标帧率
  lazy_undistortion: true         # 仅Tag局部去畸变
```

## 废弃功能清单

以下功能已从新架构中移除：
- ❌ 动态相机mode判断
- ❌ 每帧单相机PnP外参求解
- ❌ ExtrinsicSmoother外参平滑
- ❌ 运行时实时外参更新
- ❌ 单相机独立estimate_extrinsic_from_floor_tags

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 直接运行，自动处理标定
python tracker_main.py

# 查看配置
python tracker_main.py --show-config

# 强制重新标定
python tracker_main.py --force-recalib
```

## 技术细节

### 时序同步机制
- **统一时间基准**：所有相机使用`time.time_ns()`系统时间戳
- **注意**：Picamera2的SensorTimestamp是单调时钟，与系统时钟不在同一基准，无法与USB相机时间戳直接对比
- **匹配策略**：25ms窗口内多相机帧组合成同步帧组
- **精度**：软件时间戳精度足够标定使用（ms级）

### 标定策略
- **校验模式**：开机采集10组，快速验证误差
- **标定模式**：采集30组，全局BA优化所有相机RT
- **持久化**：标定结果自动保存，下次启动直接复用

### 性能优化
- 标定阶段：多线程高频采集
- 运行阶段：固定外参，无标定开销
- 检测优化：仅Tag局部ROI去畸变
- 可选功能：BEV视图可配置关闭
