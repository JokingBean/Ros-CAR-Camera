# ROS-Camera — 三相机小车追踪系统

三台固定相机（PiCamera + USB1 + USB2）协同追踪贴有 AprilTag 的移动小车。
基于 PnP 位姿估计 + GSD 加权融合，定位精度 ±2cm。

## 系统架构

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ PiCam    │   │  USB1    │   │  USB2    │
  │ 1332x990 │   │2048x1536 │   │2560x1440 │
  │ (树莓派) │   │ (树莓派) │   │ (本机PC) │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │ Tag 检测 + PnP 位姿估计
                      ▼
              ┌──────────────┐
              │  GSD 加权融合 │
              │  + 立方体中心 │
              │    校正       │
              └──────┬───────┘
                     │
                     ▼
              最终小车位置 + 车头朝向
```

## 相机参数

| 参数 | PiCam | USB1 | USB2 |
|------|-------|------|------|
| 位置 | 树莓派 CSI | 树莓派 USB | 本机 PC USB |
| 分辨率 | 1332×990 | 2048×1536 | 2560×1440 |
| 内参 fx/fy | 1064.8 / 1056.9 | 1610.3 / 1599.8 | 1997.6 / 2004.4 |
| 内参误差 | 0.049 px | 0.133 px | 0.163 px |
| 外参误差 | 1.35 px | — | 8.26 px |
| 部署高度 | ~130 cm | ~128 cm | ~132 cm |

## 小车 Tag 布局

```
        立方体 25cm × 25cm × 25cm
                Tag1 (正面, ID=1)
                  ●
          ┌───────┼───────┐
  Tag0 ●  │       │       │  ● Tag2
  (左面)  │    中心 ●     │  (右面)
          │               │
          └───────┼───────┘
                  ●
                Tag3 (背面, ID=3)

  Tag 边长: 0.135m
  车头方向: Tag1 → Tag3 (北偏东为正)
```

## 快速开始 — 一键测试

```bash
# 前置: 树莓派开机联网, 本机接 USB2
pip install -r requirements.txt paramiko
python run_test.py
```

自动完成：树莓派拍图 → 本机拍图 → 检测融合 → 生成报告。
报告在 `tracking_run_YYYYMMDD_HHMMSS/cart_tracking_report.html`。

## 首次部署 — 标定流程

### 1. 内参标定（每台相机做一次）

```bash
cd calibration_toolkit

# 树莓派上运行:
python calibrate_intrinsics.py --camera picam    # PiCamera
python calibrate_intrinsics.py --camera usb      # USB1

# 本机 PC 运行:
python calibrate_intrinsics.py --camera usb2     # USB2
```

棋盘格 9×9 内角点，方格 2cm。手持棋盘格在不同位置/角度展示，按 `s` 保存，≥15 张后按 `q` 计算。  
标定结果自动打印，填入 `config.yaml`。

### 2. 外参标定（每台相机做一次）

```bash
# 树莓派上（需要 floor_tags.yaml 和地面 Tag）:
python calibrate_extrinsics.py --camera picam --mode apriltag
python calibrate_extrinsics.py --camera usb --mode apriltag

# 本机 PC:
python calibrate_extrinsics.py --camera usb2 --mode apriltag \
    --intrinsics ../camera_calibration_usb2.json
```

相机对准地面 AprilTag，按 `c` 一键 PnP 标定。  
⚠️ **标定前确保小车 Tag (0,1,2,3) 不在视野内**——会和地面 Tag 撞号。

### 3. 填写配置

将标定结果填入 `config.yaml` 和 `extrinsics.yaml`。

## 融合原理

```
各相机独立检测小车 Tag
    │
    ▼
solvePnP → Tag 世界位置 + 车头朝向
    │
    ▼
立方体中心校正 (面偏移 ±12.5cm)
    │
    ▼
GSD 加权融合: weight = 1/GSD / Σ(1/GSD)
    │
    ▼
最终位置 + 朝向
```

- **GSD** (Ground Sampling Distance): 地面每像素对应多少毫米。越小越精确。
- **立方体中心校正**: Tag 贴在立方体侧面，Tag1(前) −12.5cm、Tag3(后) +12.5cm 得几何中心。
- **朝向融合**: 同样 GSD 加权。

## 目录结构

```
├── run_test.py              # 一键测试脚本
├── cart_report.py           # 分析报告生成（每次独立文件夹）
├── tracker.py               # GSD加权融合 + 立方体中心
├── camera_reader.py         # 相机驱动（PiCam + USB）
├── detector.py              # AprilTag 检测
├── calibrator.py            # PnP 外参标定
├── main.py                  # 实时追踪主程序
│
├── config.yaml              # 三相机内参
├── floor_tags.yaml          # 地面 Tag 世界坐标
├── extrinsics.yaml          # 外参标定结果
├── requirements.txt
│
├── bev_3cam_fusion.py       # 三相机 BEV 分析
├── bev_fusion.py            # 双相机 BEV
│
├── calibration_toolkit/     # 标定脚本 + 手册
├── bev_result/              # 双相机 BEV 结果
├── bev_3cam_result/         # 三相机 BEV 结果
└── tracking_run_*/          # 每次测试的独立输出文件夹
```

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 树莓派连不上 | 网络/IP变化 | 检查 `run_test.py` 中 `PI_HOST` |
| USB2 画面全黑 | DSHOW 驱动/延长线 | 重新插拔USB，或换 idx=0/1 |
| 外参误差 >50px | 小车Tag(0-3)和地面Tag撞号 | 标定时把小车移出视野 |
| 三个朝向不一致 | Tag面朝向定义反了 | 检查 `tracker.py` 中 `_HEADING_IN_TAG_FRAME` |
| PiCamera 颜色偏蓝 | RGB/BGR 通道反了 | `camera_reader.py` 中 `capture_array()` 不应做 `cvtColor` |
