# 快速入门指南

## 新架构说明

这是基于**静态相机+全局BA标定**的全新架构，核心改动：

- ✅ 所有相机固定安装，无动态移动
- ✅ 启动时自动外参校验+重标定
- ✅ 运行时固定外参，不再逐帧更新
- ✅ 多线程时序同步采集（25ms窗口）
- ✅ 全局BA优化（基于multi-cam-apriltag-calib）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置文件

### 1. 相机内参配置
编辑 `config/cameras_config.yaml`：
- 设置相机类型（USB/Picamera）
- 配置分辨率、帧率
- 设置内参矩阵和畸变系数

### 2. 地面Tag布局
编辑 `config/floor_tag_layout.yaml`：
- 设置地面Tag世界坐标
- 至少需要8个地面Tag用于标定

## 首次运行

```bash
# 直接运行，自动标定
python tracker_main.py

# 首次运行流程：
# 1. 未检测到外参文件
# 2. 采集30组同步帧
# 3. 执行全局BA标定
# 4. 保存 config/global_extrinsics.yaml
# 5. 进入目标追踪
```

## 日常运行

```bash
# 正常运行
python tracker_main.py

# 运行流程：
# 1. 加载历史外参
# 2. 采集10组同步帧校验
# 3. 误差 ≤ 1.2px：直接追踪
# 4. 误差 > 1.2px：自动重新标定
```

## 强制重新标定

```bash
# 相机位置改变后
python tracker_main.py --force-recalib
```

## 查看配置

```bash
python tracker_main.py --show-config
```

## 目录结构

```
ROS-Camera/
├── camera_io/              # 相机IO系统
│   └── camera_system.py    # 多线程同步采集
├── calibration/            # 标定模块
│   ├── calib_check.py      # 外参校验
│   └── auto_ba_calib.py    # 全局BA标定
├── detection/              # 检测模块
│   └── apriltag_detector.py # AprilTag检测
├── fusion_tracking/        # 融合追踪
│   └── multi_camera_fusion.py # 多观测融合
├── visualization/          # 可视化
│   └── stitch_view.py      # BEV视图
├── config/                 # 配置文件
│   ├── cameras_config.yaml
│   ├── floor_tag_layout.yaml
│   └── global_extrinsics.yaml (自动生成)
├── logs/                   # 日志
└── tracker_main.py         # 主程序

```

## 配置参数调整

### 时序同步
```yaml
timing:
  max_frame_diff_ms: 25     # 降低提高同步严格度
```

### 校验阈值
```yaml
calibration_check:
  max_allowed_repro_error: 1.2  # 提高则降低重标定频率
  check_frame_num: 10           # 校验采集帧数
  auto_capture_count: 30        # 标定采集帧数
```

### 融合模式
```yaml
fusion:
  mode: "best_select"         # 或 "weighted_average"
```

### 性能优化
```yaml
runtime:
  enable_bev_view: false      # 关闭BEV节省CPU
  target_fps: 20              # 降低FPS节省资源
  lazy_undistortion: true     # 局部去畸变
```

## 故障排查

### 标定失败
- 检查地面Tag数量（至少8个）
- 确保Tag清晰可见
- 增加采集帧数

### 时序不同步
- 检查相机硬件性能
- 增大 `max_frame_diff_ms`
- 使用相同类型相机

### 追踪抖动
- 调整平滑参数 `alpha`
- 切换融合模式
- 检查相机稳定性

## 与旧版本区别

### 废弃功能
- ❌ 动态相机支持
- ❌ 逐帧PnP外参更新
- ❌ ExtrinsicSmoother
- ❌ 运行时实时标定

### 新增功能
- ✅ 启动时外参校验
- ✅ 自动重标定触发
- ✅ 多线程时序同步
- ✅ 持久化外参管理
