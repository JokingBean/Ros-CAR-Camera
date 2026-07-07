#!/usr/bin/env python3
"""
PC 端 GUI 追踪显示器 (PyQt5)
============================
实时显示车辆位置、车头方向(tag3)、移动轨迹。
用法: python drivers/live_gui.py
"""

import socket
import json
import sys
import os
import threading
from collections import deque

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                          QPolygonF, QPainterPath, QRadialGradient)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 世界坐标参数 ----
X_MIN, X_MAX = 0.0, 4.5
Y_MIN, Y_MAX = -0.5, 5.0
GRID_STEP = 0.5

# ---- 颜色 ----
C_BG = QColor(22, 22, 38)
C_GRID = QColor(40, 40, 60)
C_GRID_TEXT = QColor(70, 70, 100)
C_BORDER = QColor(60, 60, 90)
C_CAR = QColor(255, 200, 40)
C_ARROW = QColor(50, 180, 255)
C_TRAIL = QColor(255, 200, 40, 120)
C_TAG = QColor(80, 255, 120)
C_TAG_SMALL = QColor(80, 255, 120, 60)
C_TEXT = QColor(210, 210, 220)
C_PANEL_BG = QColor(30, 30, 50)
C_LABEL = QColor(140, 140, 170)


class TrackerView(QWidget):
    """俯视追踪画布。"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 500)
        self.trail = deque(maxlen=300)
        self.raw_tags = []
        self.car_x = -99.0
        self.car_y = -99.0
        self.car_yaw = 0.0
        self.ppm = 60  # pixels per meter, 自适应缩放

    def set_data(self, car_x, car_y, car_yaw, trail, raw_tags):
        self.car_x = car_x
        self.car_y = car_y
        self.car_yaw = car_yaw
        self.trail = trail
        self.raw_tags = raw_tags

    def _world_to_view(self, x, y):
        w = self.width()
        h = self.height()
        ppm_x = (w - 80) / (X_MAX - X_MIN)
        ppm_y = (h - 80) / (Y_MAX - Y_MIN)
        ppm = min(ppm_x, ppm_y)
        off_x = (w - (X_MAX - X_MIN) * ppm) / 2
        off_y = (h - (Y_MAX - Y_MIN) * ppm) / 2
        u = off_x + (x - X_MIN) * ppm
        v = h - off_y - (y - Y_MIN) * ppm
        self.ppm = ppm
        return u, v, ppm

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 背景
        p.fillRect(0, 0, w, h, C_BG)

        _, _, ppm = self._world_to_view(0, 0)
        ox, oy = self._get_offset()
        off_x = ox
        off_y = oy

        # 网格
        pen_grid = QPen(C_GRID, 1)
        p.setPen(pen_grid)
        font_small = QFont("Consolas", 8)
        p.setFont(font_small)

        x = X_MIN
        while x <= X_MAX + 0.001:
            u, _, _ = self._world_to_view(x, Y_MIN)
            v_top, _, _ = self._world_to_view(x, Y_MIN)
            _, v_bot, _ = self._world_to_view(x, Y_MAX)
            p.drawLine(int(u), int(v_top), int(u), int(v_bot))
            p.setPen(QPen(C_GRID_TEXT))
            p.drawText(QRectF(u - 20, h - off_y + 2, 40, 14), Qt.AlignCenter, f"{x:.1f}")
            p.setPen(pen_grid)
            x += GRID_STEP

        y = Y_MIN
        while y <= Y_MAX + 0.001:
            u_left, _, _ = self._world_to_view(X_MIN, y)
            u_right, _, _ = self._world_to_view(X_MAX, y)
            _, v, _ = self._world_to_view(X_MIN, y)
            p.drawLine(int(u_left), int(v), int(u_right), int(v))
            p.setPen(QPen(C_GRID_TEXT))
            p.drawText(QRectF(off_x - 38, v - 7, 34, 14), Qt.AlignRight | Qt.AlignVCenter, f"{y:.1f}")
            p.setPen(pen_grid)
            y += GRID_STEP

        # 边框
        u0, v0, _ = self._world_to_view(X_MIN, Y_MIN)
        u1, v1, _ = self._world_to_view(X_MAX, Y_MAX)
        p.setPen(QPen(C_BORDER, 2))
        p.drawRect(QRectF(u0, v1, u1 - u0, v0 - v1))

        # Tag 位置（小绿点）
        for t in self.raw_tags:
            tx, ty = t.get("center_xy", [0, 0])
            u, v, _ = self._world_to_view(tx, ty)
            p.setBrush(C_TAG_SMALL)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(u, v), 3, 3)

        # 轨迹
        if len(self.trail) >= 2:
            path = QPainterPath()
            first = True
            alpha_step = 180 / max(len(self.trail), 1)
            for i, (tx, ty) in enumerate(self.trail):
                u, v, _ = self._world_to_view(tx, ty)
                if first:
                    path.moveTo(u, v)
                    first = False
                else:
                    path.lineTo(u, v)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 200, 40, 80), 2))
            p.drawPath(path)

        # 车位置
        if self.car_x != -99 and self.car_y != -99:
            u, v, ppm = self._world_to_view(self.car_x, self.car_y)

            # 光晕
            gradient = QRadialGradient(u, v, 18)
            gradient.setColorAt(0, QColor(255, 200, 40, 80))
            gradient.setColorAt(1, QColor(255, 200, 40, 0))
            p.setBrush(gradient)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(u, v), 18, 18)

            # 车身圆 — 车尾（橙红色）
            p.setBrush(QColor(240, 80, 60))
            p.setPen(QPen(QColor(0, 0, 0, 100), 2))
            p.drawEllipse(QPointF(u, v), 10, 10)

            # 车头 — 大箭头
            arrow_len = 35
            dx = arrow_len * np.cos(self.car_yaw)
            dy = -arrow_len * np.sin(self.car_yaw)
            tip = QPointF(u + dx, v + dy)

            # 箭头身体
            wing = 14
            base_w = 8
            nx, ny = -np.sin(self.car_yaw), -np.cos(self.car_yaw)
            base_center = QPointF(u - dx * 0.1, v - dy * 0.1)
            left = QPointF(base_center.x() + nx * base_w, base_center.y() - ny * base_w)
            right = QPointF(base_center.x() - nx * base_w, base_center.y() + ny * base_w)

            arrow = QPolygonF([tip, left, right])
            p.setBrush(QColor(50, 180, 255, 220))
            p.setPen(QPen(QColor(255, 255, 255, 180), 2))
            p.drawPolygon(arrow)

            # 车头尖端点
            p.setBrush(QColor(255, 255, 255))
            p.setPen(Qt.NoPen)
            p.drawEllipse(tip, 4, 4)

            # FRONT 标签
            p.setPen(QPen(QColor(255, 255, 255)))
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            label_u = u + dx * 1.3
            label_v = v + dy * 1.3
            p.drawText(QRectF(label_u - 25, label_v - 10, 50, 20), Qt.AlignCenter, "FRONT")

        p.end()

    def _get_offset(self):
        w = self.width()
        h = self.height()
        ppm_x = (w - 80) / (X_MAX - X_MIN)
        ppm_y = (h - 80) / (Y_MAX - Y_MIN)
        ppm = min(ppm_x, ppm_y)
        off_x = (w - (X_MAX - X_MIN) * ppm) / 2
        off_y = (h - (Y_MAX - Y_MIN) * ppm) / 2
        return off_x, off_y


class InfoPanel(QFrame):
    """右侧状态面板。"""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(200)
        self.setStyleSheet("background: #1e1e32; border-radius: 6px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self.title = QLabel("CAR TRACKER")
        self.title.setStyleSheet("color: #e94560; font: bold 14px 'Consolas';")
        layout.addWidget(self.title)

        layout.addWidget(self._sep())

        self.lbl_xy = self._make_label("XY: --")
        self.lbl_heading = self._make_label("HDG: --")
        self.lbl_grid = self._make_label("GRID: --")
        self.lbl_err = self._make_label("ERR: --")
        self.lbl_fps = self._make_label("FPS: --")
        self.lbl_cam = self._make_label("CAM: --")
        self.lbl_obs = self._make_label("OBS: --")
        self.lbl_status = self._make_label("STATUS: --")

        for w in [self.lbl_xy, self.lbl_heading, self.lbl_grid, self.lbl_err,
                   self.lbl_fps, self.lbl_cam, self.lbl_obs]:
            layout.addWidget(w)

        layout.addWidget(self._sep())
        layout.addWidget(self.lbl_status)

        self.lbl_tags = QLabel("")
        self.lbl_tags.setStyleSheet("color: #50ff78; font: 8px 'Consolas';")
        self.lbl_tags.setWordWrap(True)
        layout.addWidget(self.lbl_tags)

        layout.addStretch()

    def _sep(self):
        s = QFrame()
        s.setFrameShape(QFrame.HLine)
        s.setStyleSheet("color: #3a3a5a;")
        return s

    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #d2d2dc; font: 11px 'Consolas';")
        return lbl

    def update_info(self, data):
        x, y = data.get("x", -99), data.get("y", -99)
        yaw = data.get("yaw", 0)
        fps = data.get("fps", 0)
        err = data.get("err_cm", 0)
        gx = data.get("grid_x", -99)
        gy = data.get("grid_y", -99)
        n_cams = data.get("n_cams", 0)
        n_obs = data.get("n_obs", 0)
        raw = data.get("raw", [])

        if x != -99:
            self.lbl_xy.setText(f"XY: ({x:.3f}, {y:.3f})")
            self.lbl_heading.setText(f"HDG: {np.degrees(yaw):.0f}\u00b0")
            self.lbl_grid.setText(f"GRID: ({gx:.1f}, {gy:.1f})")
            self.lbl_err.setText(f"ERR: {err:.1f} cm")
            self.lbl_status.setText("STATUS: TRACKING")
            self.lbl_status.setStyleSheet("color: #50ff78; font: 11px 'Consolas';")
        else:
            self.lbl_xy.setText("XY: (-99, -99)")
            self.lbl_heading.setText("HDG: --")
            self.lbl_grid.setText("GRID: --")
            self.lbl_err.setText("ERR: --")
            self.lbl_status.setText("STATUS: NO DETECT")
            self.lbl_status.setStyleSheet("color: #ff5050; font: 11px 'Consolas';")

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_cam.setText(f"CAM: {n_cams}/3")
        self.lbl_obs.setText(f"OBS: {n_obs}")

        # Tag 列表
        if raw:
            tag_ids = sorted(set(t["tag_id"] for t in raw))
            cams = sorted(set(t.get("camera", "?") for t in raw))
            self.lbl_tags.setText(f"Tags: {tag_ids}\nFrom: {cams}")
        else:
            self.lbl_tags.setText("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Tracker — PyQt5")
        self.setStyleSheet("background: #161626;")

        central = QWidget()
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(10)

        self.view = TrackerView()
        hbox.addWidget(self.view, 1)

        self.panel = InfoPanel()
        hbox.addWidget(self.panel)

        self.resize(900, 650)

        # 数据
        self.latest_data = {"x": -99, "y": -99, "yaw": 0, "fps": 0}
        self.trail = deque(maxlen=300)
        self.raw_cache = []
        self.lock = threading.Lock()

        # TCP 线程
        self.tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        self.tcp_thread.start()

        # 刷新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)  # ~30fps

        self.setWindowTitle("Car Tracker — PyQt5  |  Listening :9527")

    def _tcp_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 9527))
        server.listen(1)
        server.settimeout(1.0)

        while True:
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue

            buf = ""
            try:
                while True:
                    try:
                        chunk = client.recv(4096)
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if line.strip():
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            with self.lock:
                                self.latest_data = data
                                x, y = data.get("x", -99), data.get("y", -99)
                                if x != -99:
                                    self.trail.append((x, y))
                                self.raw_cache = data.get("raw", [])
            except Exception:
                pass
            finally:
                client.close()

    def _refresh(self):
        with self.lock:
            data = dict(self.latest_data)
            trail = list(self.trail)
            raw = list(self.raw_cache)

        self.view.set_data(
            data.get("x", -99),
            data.get("y", -99),
            data.get("yaw", 0),
            trail,
            raw
        )
        self.view.update()
        self.panel.update_info(data)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
