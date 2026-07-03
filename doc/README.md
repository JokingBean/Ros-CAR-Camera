# ROS-Camera 多相机 BEV 系统 — 技术文档

## 项目概述

三台 USB 摄像头 → 实时抓图 → 外参标定 → BEV 俯视图融合 → 立方体定位精度测试

所有相机连接在一台树莓派上（100.126.101.5），通过 SSH 远程抓图，PC 端完成所有计算和显示。

## 硬件

| 项目 | 参数 |
|------|------|
| 相机 | 3 × USB 摄像头, 2560×1440, MJPG |
| 主机 | 树莓派 (100.126.101.5), pi/alcht0 |
| 计算 | PC (Windows 10), python 3.13 |
| 立方体 | 25cm 边长, 4 个面贴 AprilTag (ID 0-3, 边长 13.5cm) |
| 地面 | AprilTag 网格 (~110 个, Tag 36h11, 边长 9cm) |

## 文件结构

```
UwbCamera/
│
├── bev_generic.py          # BEV 引擎（核心）
│   ├── BevGenerator 类
│   ├── _make_bev_undistorted()    # 外参投影（画面平滑）
│   ├── _make_bev_from_homography() # homography 回退
│   ├── _fuse_bevs()               # 最近相机加权融合
│   └── save_report()              # HTML 报告生成
│
├── run_all.py              # 一键全流程
│   抓图 → 标定 → BEV → 报告
│
├── calibrate_all.py        # 交互式外参标定
│   (homography 初始化 + solvePnP)
│
├── precision_test.py       # 立方体定位精度测试
│   # 抓图 → 检测立方体 Tag → GSD 加权融合 → 吸附网格 → 误差分析
│
├── gui_app.py              # Tkinter GUI 控制台
│   动态相机复选框、实时 BEV、标定、追踪
│
├── live_tracker.py         # 低延时实时追踪
│
├── camera_reader.py        # USB 相机读取封装
│
├── config.yaml             # 相机配置（内参、分辨率、host）
├── extrinsics.yaml         # 外参数据（homography-init solvePnP）
├── floor_tags.yaml         # 地面 AprilTag 世界坐标
├── calibrator.py           # PnP 标定核心
├── tracker.py / detector.py # Tag 检测 + 3D 追踪
│
└── calibration_toolkit/    # 棋盘格内参标定工具
```

## 数据流

```
Pi 抓图 (SSH) ─→ PC
                    │
           ┌────────┴────────┐
           │  BEV (run_all)   │   精度测试 (precision_test)
           │                  │   │
     CLAHE → AprilTag         │   │
           │      │           │   │
     外参投影 (K,R,t)         │   │  solvePnP → Tag 3D XY
           │                  │   │
     最近相机加权融合          │   │  margin ≥ 20 过滤
           │                  │   │
     BEV 图 + HTML 报告       │   中位数共识 → 误差分析
```

### BEV 模式（run_all.py → bev_generic.py）

1. SSH 三台相机抓图（亮度 BRIGHTNESS=30, CONTRAST=40, GAMMA=100 统一）
2. CLAHE 局部对比度增强
3. 外参投影（_make_bev_undistorted）优先 —— 经过去畸变，画面平滑
4. 若外参覆盖 < 500px 或外参无效，回退到 homography
5. 最近相机加权融合 —— 重叠区优先用距离最近的相机
6. 标定 → 保存 extrinsics.yaml → 生成 HTML 报告

### 精度测试模式（precision_test.py）

1. SSH 三台相机抓图
2. 每台相机 detect_cube_extrinsics() —— solvePnP 求 Tag 3D 世界坐标
3. Tag XY 直接作为立方体中心 XY（面偏移 12.5cm 因法线方向精度不够已去掉）
4. 过滤 margin < 20 的不可靠检测
5. 中位数共识作为最终位置
6. 自动吸附到最近 0.5m 网格点作为真值
7. 计算误差 → 保存到 precision_runs/ 文件夹

### 外参标定（run_all.py / calibrate_all.py）

1. 从地面 Tag 计算 homography
2. 从 H 提取旋转矩阵 R 作为初始值
3. solvePnP 微调 (R, t)
4. 所有地面 Tag 参与，homography 初始化保证收敛到合理高度

## 关键算法

### Homography 到外参

H = λ · K · [r1 r2 t]

```
K⁻¹ · H → [r1 r2 t] → R = [r1 r2 r1×r2] → solvePnP 微调
```

### 外参投影

```
P_cam = R · P_world + t
u = K[0,0] · x_cam/z_cam + K[0,2]
v = K[1,1] · y_cam/z_cam + K[1,2]
```

### 最近相机加权融合

对每个 BEV 像素 (u, v) → 世界坐标 (x_w, y_w) → 投影到各相机 → 取最近相机的颜色。

权重 = 1 / (1 + 到相机 BEV 中心的像素距离 / (BEV 对角线 / 4))

### 立方体定位

```
solvePnP(Tag 四角点, K, dist) → R_tag2cam, t_tag2cam
Tag 世界坐标: tw = R_c2w @ t_tag2cam + t_c2w
立方体中心 XY = Tag XY（面偏移已省略）
融合: median({所有相机关联的所有 Tag 的 XY})
```

## 精度表现

| 条件 | 典型误差 |
|------|:--:|
| 3 相机 + 多面 Tag + margin≥20 | < 2cm |
| 2 相机 + 单面 Tag | 3-5cm |
| 1 相机 | 5-15cm |
| usb3 低可信度 (margin<20) | 过滤 |

## 配置文件

### config.yaml 相机定义

```yaml
cameras:
  - name: usb1      # Pi, /dev/video0
    type: usb; host: pi; device: "0"
    resolution: [2560, 1440]; fps: 30
    camera_matrix: {fx: 1997.6, fy: 2004.4, cx: 1203.9, cy: 784.2}
    dist_coeffs: [0.08367, -0.15649, 0.00321, -0.00835, 0.11271]

  - name: usb2      # Pi, /dev/video2
    ... 相同内参 ...

  - name: usb3      # Pi, /dev/video4
    ... 相同内参 ...
```

### extrinsics.yaml

```yaml
usb1:
  R: [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
  t: [tx, ty, tz]
```

## 使用方式

```bash
# 一键全流程
python run_all.py

# 精度测试
python precision_test.py
# 按 Enter 测量, 's' 看统计, 'q' 退出

# GUI 控制台
python gui_app.py

# 交互式标定
python calibrate_all.py
# 按 'c' 采集标定, 's' 保存, 'q' 退出
```

## 已知限制

1. **三台共享同一套内参** — 每台相机实际焦距有差异，导致 usb1/usb2/usb3 定位存在系统偏差。需要每台单独棋盘格标定才能进 2cm。
2. **面偏移已省略** — Tag 在立方体面上距中心 12.5cm，但目前解算的法线方向不够准，去掉偏移后用中位数反而更稳定。
3. **usb3 亮度偏低** — idx=4 的相机信号弱，需提高 BRIGHTNESS 或增加 Gain。
4. **不可靠检测过滤** — margin < 20 的观测自动丢弃，避免拉低精度。
