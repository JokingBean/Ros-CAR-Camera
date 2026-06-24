# ROS-Camera — Multi-Camera AprilTag Tracking System

Dual-camera (PiCamera + USB) 3D object tracking using AprilTags and PnP-based extrinsic calibration.
Designed for static deployment — calibrate once, track continuously.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      main.py                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ camera   │  │ detector │  │calibrator│  │ tracker  │ │
│  │ _reader  │  │ .py      │  │ .py      │  │ .py      │ │
│  │ .py      │  │          │  │          │  │          │ │
│  │ PiCam+USB│  │AprilTag  │  │PnP+RANSAC│  │Multi-cam │ │
│  │ capture  │  │ detect   │  │ extrinsic│  │ 3D fusion│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Flow**: Open cameras (PiCam first) → load/check extrinsics → detect AprilTags → PnP pose → multi-camera fusion → display

## Cameras

| | PiCamera | USB |
|---|---|---|
| Sensor | IMX477 (CSI) | 2K USB Camera (2bdf:0281) |
| Resolution | 1332 × 990 @ 60fps | 2048 × 1536 @ 25fps |
| Intrinsic calib. | 33 images, **0.049 px** | 26 images, **0.133 px** |
| Intrinsic fx/fy | 1064.81 / 1056.90 | 1610.26 / 1599.84 |
| Extrinsic (PnP) | **1.35 px** reproj | 23.27 px reproj |
| Height validated | 131 cm (target 130) | 128 cm (target 130) |

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Calibrate (first time only)
cd calibration_toolkit
python calibrate_intrinsics.py --camera picam   # PiCamera first
python calibrate_intrinsics.py --camera usb     # USB second
python calibrate_extrinsics.py --camera picam --mode apriltag
python calibrate_extrinsics.py --camera usb --mode apriltag

# 3. Run
cd ..
python main.py
```

## BEV Analysis

Two-camera bird's-eye view fusion with per-tag GSD (Ground Sampling Distance) quality assessment.

| Metric | PiCam | USB |
|---|---|---|
| Floor coverage | 60.5% | 44.3% |
| Overlap tags | 40 (theoretical) | 11 (detected) |
| Overlap GSD | 3.7 mm/px | **2.4 mm/px** |
| Better in overlap | 0 tags | **40 tags** |

**USB is 56% finer** than PiCam in the overlapping zone — 2048×1536 packs more pixels per ground meter.

See `bev_result/bev_report.html` for the full interactive report with tag-by-tag GSD comparison.

## Directory

```
├── main.py                 # Entry point
├── camera_reader.py        # PiCamera + USB capture (MJPG, platform-aware)
├── detector.py             # pupil-apriltags wrapper
├── calibrator.py           # PnP extrinsic calibration + persistence
├── tracker.py              # Multi-camera 3D pose fusion
├── config.yaml             # Camera intrinsics + runtime settings
├── floor_tags.yaml         # 110 floor AprilTag world coordinates
├── extrinsics.yaml         # Calibrated camera extrinsics
│
├── bev_fusion.py           # BEV projection + GSD analysis
├── tag_overlap_compare.py  # Side-by-side tag comparison
├── analyze_snapshots.py    # Snapshot pair analysis
│
├── bev_result/             # BEV results: report, images, HTML
├── calibration_toolkit/    # Intrinsic + extrinsic calibration scripts
└── requirements.txt
```

## Key Design Decisions

- **Static cameras** — extrinsics calibrated once, fixed during tracking
- **PiCamera initialized first** — avoids V4L2/libcamera driver contention on Raspberry Pi
- **MJPG format for USB** — required for resolutions > 640×480
- **PnP + RANSAC** — robust extrinsic calibration from floor AprilTags
- **GSD metric** — quantifies per-tag localization precision between cameras

## License

MIT
