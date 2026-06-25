#!/usr/bin/env python3
"""
ROS-Camera GUI — 三相机小车追踪控制台
======================================
左侧: 摄像头列表 + 状态面板
中间: BEV 融合俯视图（背景）
右侧: 功能按键
底部: 日志输出
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import time
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import cv2
import yaml
import numpy as np
from PIL import Image, ImageTk

# ==============================================================
# 全局状态
# ==============================================================
APP_TITLE = "ROS-Camera 三相机小车追踪控制台"

# 相机配置
CAMERA_CONFIGS = {
    "picam_1":  {"name": "PiCam",     "type": "picamera", "device": "picamera:0", "res": "1332x990",   "location": "Pi"},
    "usb_cam_1": {"name": "USB1",     "type": "usb",      "device": "0",          "res": "2048x1536", "location": "Pi"},
    "usb_cam_2": {"name": "USB2",     "type": "usb",      "device": "1",          "res": "2560x1440", "location": "Local"},
}

PI_HOST = "100.101.225.34"
PI_USER = "pi"
PI_PASS = "alcht0"

# ==============================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1280x800")
        root.configure(bg="#1a1a2e")

        # 状态
        self.bev_image = None       # 当前 BEV PIL Image
        self.tracking_running = False
        self.camera_status = {}     # {cam_key: True/False}
        self.log_queue = queue.Queue()

        self._build_ui()
        self._process_log_queue()

        # 启动后自动检测摄像头
        self.root.after(500, self.refresh_cameras)

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self):
        # --- 顶部标题栏 ---
        title_bar = tk.Frame(self.root, bg="#0f3460", height=40)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(title_bar, text=APP_TITLE, bg="#0f3460", fg="white",
                 font=("Microsoft YaHei", 14, "bold")).pack(side=tk.LEFT, padx=16, pady=5)
        tk.Label(title_bar, text="v2.0", bg="#0f3460", fg="#888",
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=16, pady=5)

        # --- 主面板 ---
        main_panel = tk.Frame(self.root, bg="#1a1a2e")
        main_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 左侧: 摄像头列表
        self._build_camera_panel(main_panel)

        # 中间: BEV 显示
        self._build_bev_panel(main_panel)

        # 右侧: 控制面板
        self._build_control_panel(main_panel)

        # 底部: 日志
        self._build_log_panel()

    def _build_camera_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" 摄像头 ", bg="#16213e", fg="#e0e0e0",
                              font=("Microsoft YaHei", 10), width=220)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4))
        frame.pack_propagate(False)

        # 刷新按钮
        btn_frame = tk.Frame(frame, bg="#16213e")
        btn_frame.pack(fill=tk.X, padx=6, pady=6)
        self.refresh_btn = tk.Button(btn_frame, text="刷新检测", command=self.refresh_cameras,
                                      bg="#0f3460", fg="white", font=("Microsoft YaHei", 9),
                                      relief=tk.FLAT, padx=12, pady=4)
        self.refresh_btn.pack(fill=tk.X)

        # 分隔
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)

        # Pi 分组
        tk.Label(frame, text="树莓派 (100.101.225.34)", bg="#16213e", fg="#e17055",
                 font=("Microsoft YaHei", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(8,2))
        self.pi_frame = tk.Frame(frame, bg="#16213e")
        self.pi_frame.pack(fill=tk.X, padx=6)
        self.pi_labels = {}

        # 本地分组
        tk.Label(frame, text="本机 PC", bg="#16213e", fg="#55efc4",
                 font=("Microsoft YaHei", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(12,2))
        self.local_frame = tk.Frame(frame, bg="#16213e")
        self.local_frame.pack(fill=tk.X, padx=6)
        self.local_labels = {}

        # 状态提示
        self.cam_status_label = tk.Label(frame, text="等待检测...", bg="#16213e", fg="#888",
                                          font=("Microsoft YaHei", 8))
        self.cam_status_label.pack(padx=8, pady=10)

    def _build_bev_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" 融合俯视图 (BEV) ", bg="#16213e", fg="#e0e0e0",
                              font=("Microsoft YaHei", 10))
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.bev_canvas = tk.Canvas(frame, bg="#0a0a15", highlightthickness=0)
        self.bev_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 占位文字
        self.bev_text_id = self.bev_canvas.create_text(
            400, 350, text="点击「场地融合」生成俯视图",
            fill="#444", font=("Microsoft YaHei", 16))

    def _build_control_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" 控制 ", bg="#16213e", fg="#e0e0e0",
                              font=("Microsoft YaHei", 10), width=200)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4,0))
        frame.pack_propagate(False)

        btn_cfg = {"font": ("Microsoft YaHei", 9), "relief": tk.FLAT,
                    "padx": 10, "pady": 6, "fill": tk.X}

        # === 标定 ===
        tk.Label(frame, text="标定", bg="#16213e", fg="#e94560",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(10,4))

        self.btn_intrinsic = tk.Button(frame, text="内参标定 (选摄像头)", bg="#533a3a", fg="white",
                                        command=self.calibrate_intrinsic, **btn_cfg)
        self.btn_intrinsic.pack(padx=6, pady=2)

        self.btn_extrinsic = tk.Button(frame, text="外参标定 (自动)", bg="#533a3a", fg="white",
                                        command=self.calibrate_extrinsic, **btn_cfg)
        self.btn_extrinsic.pack(padx=6, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)

        # === 融合 ===
        tk.Label(frame, text="融合", bg="#16213e", fg="#e94560",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(4,4))

        self.btn_fusion = tk.Button(frame, text="场地融合 (BEV)", bg="#2d5a2d", fg="white",
                                     command=self.do_fusion, **btn_cfg)
        self.btn_fusion.pack(padx=6, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)

        # === 追踪 ===
        tk.Label(frame, text="追踪", bg="#16213e", fg="#e94560",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(4,4))

        self.btn_track = tk.Button(frame, text="小车定位 (单次)", bg="#2d2d5a", fg="white",
                                    command=self.do_tracking_once, **btn_cfg)
        self.btn_track.pack(padx=6, pady=2)

        self.btn_track_live = tk.Button(frame, text="实时追踪 (开始)", bg="#2d2d5a", fg="white",
                                         command=self.toggle_live_tracking, **btn_cfg)
        self.btn_track_live.pack(padx=6, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)

        # === 模式 ===
        tk.Label(frame, text="融合模式", bg="#16213e", fg="#e94560",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(4,2))
        self.fusion_mode = tk.StringVar(value="gsd_weighted")
        modes = [("GSD 加权", "gsd_weighted"), ("最佳选择", "best_select"), ("简单平均", "average")]
        for text, val in modes:
            tk.Radiobutton(frame, text=text, variable=self.fusion_mode, value=val,
                           bg="#16213e", fg="#ccc", selectcolor="#0f3460",
                           font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=16)

        # === 导出 ===
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        self.btn_export = tk.Button(frame, text="导出完整报告", bg="#4a3a2d", fg="white",
                                     command=self.export_report, **btn_cfg)
        self.btn_export.pack(padx=6, pady=2)

        # 图例
        tk.Label(frame, text="● PiCam  ● USB1  ● USB2  ● 小车",
                 bg="#16213e", fg="#888", font=("Microsoft YaHei", 7)).pack(pady=10)

    def _build_log_panel(self):
        self.log_text = scrolledtext.ScrolledText(self.root, height=6, bg="#0a0a15", fg="#aaa",
                                                   font=("Consolas", 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0,4))
        self.log_text.insert(tk.END, "ROS-Camera 控制台就绪\n")

    # ==================================================================
    # 日志
    # ==================================================================
    def log(self, msg):
        self.log_queue.put(msg)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_text.see(tk.END)
        self.root.after(100, self._process_log_queue)

    # ==================================================================
    # 摄像头检测
    # ==================================================================
    def refresh_cameras(self):
        """检测所有摄像头可用性"""
        self.log("正在检测摄像头...")
        self.cam_status_label.config(text="检测中...", fg="#e17055")

        def _detect():
            status = {}

            # 本地 USB 摄像头
            for idx in range(3):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        status[f"local_{idx}"] = True

            # 树莓派
            try:
                import paramiko
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=5)
                stdin, stdout, stderr = ssh.exec_command('ls /dev/video2 2>/dev/null && echo PICAM_OK; ls /dev/video0 2>/dev/null && echo USB_OK')
                out = stdout.read().decode()
                status["pi_picam"] = "PICAM_OK" in out
                status["pi_usb"] = "USB_OK" in out
                ssh.close()
                status["pi_online"] = True
            except Exception as e:
                status["pi_online"] = False
                self.log(f"树莓派连接失败: {e}")

            self.camera_status = status
            self.root.after(0, self._update_camera_list, status)

        threading.Thread(target=_detect, daemon=True).start()

    def _update_camera_list(self, status):
        # 清空
        for w in self.pi_frame.winfo_children():
            w.destroy()
        for w in self.local_frame.winfo_children():
            w.destroy()

        # Pi 摄像头
        pi_online = status.get("pi_online", False)
        if pi_online:
            tk.Label(self.pi_frame, text="在线", bg="#16213e", fg="#55efc4",
                     font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=2)
        else:
            tk.Label(self.pi_frame, text="离线", bg="#16213e", fg="#e17055",
                     font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=2)

        for key, label in [("pi_picam", "PiCamera"), ("pi_usb", "USB1")]:
            ok = status.get(key, False)
            color = "#55efc4" if ok else "#e17055"
            sym = "●" if ok else "○"
            tk.Label(self.pi_frame, text=f"  {sym} {label}", bg="#16213e", fg=color,
                     font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10)

        # 本地 USB
        local_count = sum(1 for k, v in status.items() if k.startswith("local_") and v)
        tk.Label(self.local_frame, text=f"发现 {local_count} 个", bg="#16213e", fg="#888",
                 font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=2)

        for i in range(3):
            key = f"local_{i}"
            ok = status.get(key, False)
            color = "#55efc4" if ok else "#555"
            sym = "●" if ok else "○"
            label = f"USB2 (idx={i})" if i == 1 else f"Camera idx={i}"
            tk.Label(self.local_frame, text=f"  {sym} {label}", bg="#16213e", fg=color,
                     font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10)

        self.cam_status_label.config(text="检测完成", fg="#888")
        self.log("摄像头检测完成")

    # ==================================================================
    # 功能实现（后台线程 + 回调）
    # ==================================================================
    def _run_in_thread(self, target, on_done=None):
        """后台执行，完成后回调主线程。"""
        def wrapper():
            try:
                result = target()
                if on_done:
                    self.root.after(0, lambda: on_done(result))
            except Exception as e:
                self.log(f"错误: {e}")
                import traceback
                traceback.print_exc()
        threading.Thread(target=wrapper, daemon=True).start()

    # --- 内参标定 ---
    def calibrate_intrinsic(self):
        cam_choice = tk.simpledialog.askstring("内参标定",
            "输入摄像头 (picam / usb / usb2):", parent=self.root)
        if not cam_choice or cam_choice not in ("picam", "usb", "usb2"):
            return
        self.log(f"启动内参标定: {cam_choice}")
        self._run_in_thread(
            lambda: os.system(f'cd calibration_toolkit && python calibrate_intrinsics.py --camera {cam_choice}'),
            lambda r: self.log("内参标定完成"))

    # --- 外参标定 ---
    def calibrate_extrinsic(self):
        self.log("启动外参标定 (需要小车移出视野!)")
        messagebox.showinfo("外参标定", "请确保小车 Tag (0,1,2,3) 不在任何相机视野内，\n然后对每台相机按 'c' 标定。")
        self._run_in_thread(
            lambda: os.system('cd calibration_toolkit && python calibrate_extrinsics.py --camera picam --mode apriltag'),
            lambda r: self.log("外参标定完成"))

    # --- 场地融合 ---
    def do_fusion(self):
        self.log("正在拍摄三相机图像并生成 BEV...")
        def task():
            import subprocess
            # 拍图
            subprocess.run([sys.executable, "cart_report.py"], capture_output=True, timeout=60)
            # 找最新 cart_bev.jpg
            dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
            if dirs:
                bev_path = dirs[0] / "cart_bev.jpg"
                if bev_path.exists():
                    return str(bev_path)
            return None

        def on_done(bev_path):
            if bev_path:
                self._load_bev(bev_path)
                self.log(f"BEV 已加载: {bev_path}")
            else:
                self.log("BEV 生成失败")

        self._run_in_thread(task, on_done)

    def _load_bev(self, path):
        img = Image.open(path)
        # 缩放到 canvas 大小
        cw = self.bev_canvas.winfo_width() or 700
        ch = self.bev_canvas.winfo_height() or 600
        img.thumbnail((cw, ch), Image.LANCZOS)
        self.bev_image = ImageTk.PhotoImage(img)
        self.bev_canvas.delete("all")
        self.bev_canvas.create_image(cw//2, ch//2, image=self.bev_image, anchor=tk.CENTER)
        self.bev_text_id = None

    # --- 单次定位 ---
    def do_tracking_once(self):
        self.log("正在执行单次小车定位...")
        def task():
            import subprocess
            result = subprocess.run([sys.executable, "cart_report.py"], capture_output=True, text=True, timeout=60)
            # 从输出提取位置
            for line in result.stdout.split('\n'):
                if 'Final position' in line:
                    return line.strip()
            return "定位完成"

        def on_done(msg):
            self.log(msg)
            # 刷新 BEV
            dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
            if dirs:
                bev_path = dirs[0] / "cart_bev.jpg"
                if bev_path.exists():
                    self._load_bev(str(bev_path))

        self._run_in_thread(task, on_done)

    # --- 实时追踪 ---
    def toggle_live_tracking(self):
        if self.tracking_running:
            self.tracking_running = False
            self.btn_track_live.config(text="实时追踪 (开始)", bg="#2d2d5a")
            self.log("实时追踪已停止")
        else:
            self.tracking_running = True
            self.btn_track_live.config(text="实时追踪 (停止)", bg="#5a2d2d")
            self.log("实时追踪已启动 (每2秒)")
            self._live_tracking_loop()

    def _live_tracking_loop(self):
        if not self.tracking_running:
            return
        def task():
            import subprocess
            subprocess.run([sys.executable, "cart_report.py"], capture_output=True, timeout=60)
            return True
        def on_done(_):
            if self.tracking_running:
                dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
                if dirs:
                    bev_path = dirs[0] / "cart_bev.jpg"
                    if bev_path.exists():
                        self._load_bev(str(bev_path))
                self.root.after(2000, self._live_tracking_loop)
        self._run_in_thread(task, on_done)

    # --- 导出报告 ---
    def export_report(self):
        dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
        if not dirs:
            messagebox.showwarning("导出", "没有找到报告文件夹，请先执行一次定位")
            return
        latest = dirs[0]
        report = latest / "cart_tracking_report.html"
        if report.exists():
            import webbrowser
            webbrowser.open(str(report))
            self.log(f"已打开报告: {report}")
        else:
            self.log("报告文件不存在")


# ==============================================================
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
