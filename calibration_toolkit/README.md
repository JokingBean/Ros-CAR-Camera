# 相机标定实施手册

## 概述

本工具包提供 ROS-Camera 多相机追踪系统所需的**内参标定**和**外参标定**，
支持 PiCamera (CSI) 和 USB 相机两种类型。

### 前置条件

- 树莓派已连接相机（PiCamera + USB）
- 已安装依赖：`pip install opencv-python numpy pyyaml picamera2 pupil-apriltags`
- **9×9 棋盘格**（内角点 9×9，方格边长 2cm，可自行打印）
- 地面已布置 AprilTag（参考 `floor_tags.yaml`）

---

## 一、内参标定

每台相机都必须先做内参标定，获得 `camera_matrix` 和 `distortion_coeffs`。

### 1.1 执行

```bash
# PiCamera (1332×990)
python calibrate_intrinsics.py --camera picam

# USB 相机 (2560×1440)
python calibrate_intrinsics.py --camera usb
```

### 1.2 操作流程

```
┌─────────────────────────────────────────────┐
│ 1. 手持棋盘格，放在相机视野内                │
│ 2. 棋盘格应占据画面 30%-70%                  │
│ 3. 移动棋盘格到不同位置、不同角度             │
│    ├─ 左上角、右上角、左下角、右下角          │
│    ├─ 靠近相机、远离相机                      │
│    ├─ 向左倾斜、向右倾斜、向前/后倾斜         │
│    └─ 旋转 45°、90°                          │
│ 4. 每个位置棋盘格稳定后按 's' 保存            │
│ 5. 目标: ≥15 张，推荐 20 张                  │
│ 6. 按 'q' 结束采集，自动计算并保存            │
└─────────────────────────────────────────────┘
```

### 1.3 质量检查

标定完成后，检查输出的指标：

| 指标 | 正常范围 | 异常处理 |
|------|---------|---------|
| **重投影误差** | < 0.5 px | > 0.5: 角度变化不够或棋盘格检测不准 |
| **fy/fx 比值** | 0.95 - 1.05 | 偏大/偏小: 棋盘格未覆盖足够画面区域 |
| **cx, cy** | 约为图像宽/高的一半 | 偏差大: 采集集中在画面一侧 |

### 1.4 输出

```
camera_calibration_picam.json    # PiCamera 内参
camera_calibration_usb.json      # USB 内参
```

---

## 二、外参标定

外参标定求解相机在世界坐标系中的 3D 位置和朝向。**推荐使用 AprilTag 模式**
（利用已布置的地面 Tag），棋盘格模式作为备选。

### 2.1 AprilTag 模式（推荐）

```bash
python calibrate_extrinsics.py --camera picam --mode apriltag
python calibrate_extrinsics.py --camera usb --mode apriltag
```

```
┌─────────────────────────────────────────────┐
│ 1. 确保相机视野内能看到 ≥6 个地面 AprilTag  │
│ 2. 观察实时画面，检查 Tag 检测是否正确       │
│ 3. 按 'c' 采集当前帧 → 自动 PnP + RANSAC   │
│ 4. 检查重投影误差                             │
│    ├─ < 5px: 良好                           │
│    ├─ 5-15px: 可用，建议再采集几张           │
│    └─ > 15px: 检查内参或 Tag 坐标           │
└─────────────────────────────────────────────┘
```

### 2.2 棋盘格模式（备选）

```bash
python calibrate_extrinsics.py --camera picam --mode chessboard
```

```
┌─────────────────────────────────────────────┐
│ 1. 将 9×9 棋盘格平放在地面上                 │
│ 2. 棋盘格原点 (左上角第1个角点) = 世界原点   │
│ 3. X 轴沿棋盘格行方向，Y 轴沿列方向          │
│ 4. 确认角点全部检测到（绿色）后按 'c'        │
│                                               │
│ 注意: 此方法计算的是"棋盘格坐标系"下的位姿    │
│ 如需世界坐标对齐，需额外测量棋盘格放置位置    │
└─────────────────────────────────────────────┘
```

### 2.3 输出

```
extrinsics_picam_1.yaml          # PiCamera 外参 (YAML, 多相机系统用)
extrinsics_usb_cam_1.yaml        # USB 外参
camera_extrinsic_picam_1.json    # 单应性矩阵 (旧版兼容)
```

---

## 三、部署到主项目

### 3.1 内参部署

将标定结果填入主项目 `config.yaml`:

```yaml
cameras:
  - name: picam_1
    type: picamera
    resolution: [1332, 990]
    camera_matrix:
      fx: <标定结果的 fx>
      fy: <标定结果的 fy>
      cx: <标定结果的 cx>
      cy: <标定结果的 cy>
    dist_coeffs: [<k1>, <k2>, <p1>, <p2>, <k3>]

  - name: usb_cam_1
    type: usb
    resolution: [2560, 1440]
    camera_matrix:
      fx: <标定结果的 fx>
      ...
```

### 3.2 外参部署

合并两个相机的外参文件：

```bash
# 方式1: 手动合并
# 将 extrinsics_picam_1.yaml 和 extrinsics_usb_cam_1.yaml 合并为 extrinsics.yaml

# 方式2: 直接用 Python
python -c "
import yaml
result = {}
for cam in ['picam_1', 'usb_cam_1']:
    with open(f'extrinsics_{cam}.yaml') as f:
        result.update(yaml.safe_load(f))
with open('../extrinsics.yaml', 'w') as f:
    yaml.dump(result, f, default_flow_style=None)
print('extrinsics.yaml 已生成')
"
```

---

## 四、完整标定流程（检查清单）

- [ ] **1.** 确认相机已连接，运行 `python calibrate_intrinsics.py --camera picam`
- [ ] **2.** 采集 ≥15 张棋盘格图像，检查重投影误差 < 0.5px
- [ ] **3.** 运行 `python calibrate_intrinsics.py --camera usb`
- [ ] **4.** 采集 ≥15 张棋盘格图像，检查重投影误差 < 0.5px
- [ ] **5.** 将内参填入主项目 `config.yaml`
- [ ] **6.** 运行 `python calibrate_extrinsics.py --camera picam --mode apriltag`
- [ ] **7.** 按 'c' 标定，确认重投影误差 < 10px
- [ ] **8.** 运行 `python calibrate_extrinsics.py --camera usb --mode apriltag`
- [ ] **9.** 按 'c' 标定，确认重投影误差 < 10px
- [ ] **10.** 合并外参到 `extrinsics.yaml`
- [ ] **11.** 运行主项目 `python main.py` 验证

---

## 五、故障排查

| 现象 | 可能原因 | 解决 |
|------|---------|------|
| USB 相机分辨率不是 2560×1440 | 未使用 MJPG 格式 | 确认 config 中 fourcc 为 MJPG |
| 棋盘格始终检测不到 | 光照/反光/模糊 | 调整光照角度，避免反光 |
| solvePnPRansac 失败 | 地面 Tag 少于 6 个 | 调整相机角度让更多 Tag 入镜 |
| 外参重投影误差 > 20px | 内参不准或 Tag 坐标有误 | 先重新标定内参，确认 floor_tags.yaml |
| PiCamera 打不开 | 未使用 Picamera2 初始化顺序 | 确保 PiCamera 先于 USB 打开 |
