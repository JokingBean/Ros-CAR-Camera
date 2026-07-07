#!/usr/bin/env python3
"""
PC 端定位接收器 — TCP Server
=============================
监听 TCP 端口，接收树莓派发来的定位结果，实时显示 XY + 误差 + FPS。

用法:
  python drivers/live.py [--port 9527]
"""

import socket
import json
import sys
import os
import time
import argparse
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "tracking_logs")


class PcReceiver:
    """PC 端 TCP 服务器，接收 Pi 发来的定位数据。"""

    def __init__(self, host="0.0.0.0", port=9527, log=True):
        self.host = host
        self.port = port
        self.log = log
        self.server_sock = None
        self.client_sock = None
        self.client_addr = None

        self.pos_hist = deque(maxlen=10)
        self.err_hist = deque(maxlen=100)
        self.fps_hist = deque(maxlen=50)
        self.total_updates = 0
        self.valid_updates = 0
        self.start_time = None
        self.last_print = 0

        if log:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(os.path.join(LOG_DIR, f"track_{ts}.jsonl"), "a")
            print(f"日志: {os.path.join(LOG_DIR, f'track_{ts}.jsonl')}")
        else:
            self.log_file = None

    def start_server(self):
        """启动 TCP 服务器，等待 Pi 连接。"""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(1.0)

        print(f"\n{'='*55}")
        print(f"  PC 定位接收器")
        print(f"  监听: {self.host}:{self.port}")
        print(f"  等待 Pi 连接...")
        print(f"{'='*55}")

    def wait_for_client(self):
        """等待一个客户端连接。"""
        while True:
            try:
                self.client_sock, self.client_addr = self.server_sock.accept()
                self.client_sock.settimeout(1.0)
                self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"\n[连接] Pi {self.client_addr[0]}:{self.client_addr[1]} 已连接\n")
                self.start_time = time.time()
                return True
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                return False

    def process_line(self, line):
        """处理一行 JSON 数据。"""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        x = data.get("x")
        y = data.get("y")
        fps = data.get("fps", 0)
        err = data.get("err_cm")
        n_cams = data.get("n_cams", 0)
        n_obs = data.get("n_obs", 0)
        raw_tags = data.get("raw", [])

        self.total_updates += 1

        # 记录 FPS
        self.fps_hist.append(fps)

        if x is not None and y is not None and x != -99.0:
            self.valid_updates += 1
            self.pos_hist.append([x, y])
            if err is not None:
                self.err_hist.append(err)

        # 控制台输出（每秒最多 10 次）
        now = time.time()
        if now - self.last_print > 0.1:
            self.last_print = now
            avg_fps = np.mean(self.fps_hist) if HAS_NUMPY and self.fps_hist else fps

            if x is not None:
                gx, gy = data.get("grid_x", 0), data.get("grid_y", 0)
                avg_err = np.mean(self.err_hist) if HAS_NUMPY and self.err_hist else (err or 0)

                # 相机 + Tag 信息
                tag_info = ""
                if raw_tags:
                    tag_info = " | ".join(
                        f"{t['camera']}T{t['tag_id']}" for t in raw_tags)

                uptime = time.time() - self.start_time if self.start_time else 0
                print(f"\r  XY=({x:.3f},{y:.3f})  "
                      f"grid=({gx:.1f},{gy:.1f})  "
                      f"err={err:.1f}cm(avg={avg_err:.1f}cm)  "
                      f"FPS={fps:.1f}(avg={avg_fps:.1f})  "
                      f"[{n_cams}cam, {n_obs}obs]  {int(uptime)}s  "
                      f"{tag_info[:80]:<80}",
                      end="", flush=True)
            else:
                uptime = time.time() - self.start_time if self.start_time else 0
                tag_str = f"({data.get('x',-99):.0f},{data.get('y',-99):.0f},{data.get('z',-99):.0f})"
                print(f"\r  无识别 {tag_str}  FPS={fps:.1f}  [{n_cams}cam]  {int(uptime)}s  ",
                      end="", flush=True)

        # 写入日志
        if self.log_file:
            self.log_file.write(line.strip() + "\n")
            self.log_file.flush()

    def run(self):
        """主循环：接收数据并显示。"""
        self.start_server()

        while self.wait_for_client():
            print("  接收定位数据 (Ctrl+C 停止)...\n")
            buf = ""
            try:
                while True:
                    try:
                        chunk = self.client_sock.recv(4096)
                    except socket.timeout:
                        continue
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        break

                    if not chunk:
                        print("\n[断开] Pi 已断开连接")
                        break

                    buf += chunk.decode("utf-8")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if line.strip():
                            self.process_line(line)

            except KeyboardInterrupt:
                print("\n\n停止接收")
                break
            finally:
                if self.client_sock:
                    self.client_sock.close()
                    self.client_sock = None

            print(f"\n[等待] 等待 Pi 重新连接...")
            # 打印统计
            self.print_summary()

    def print_summary(self):
        """打印汇总统计。"""
        if self.total_updates == 0:
            return
        uptime = time.time() - self.start_time if self.start_time else 0
        print(f"\n  {'─'*40}")
        print(f"  总计: {self.total_updates} 帧  "
              f"有效定位: {self.valid_updates} 帧  "
              f"运行时间: {uptime:.0f}s")
        if HAS_NUMPY:
            if self.fps_hist:
                print(f"  平均 FPS: {np.mean(self.fps_hist):.1f}  "
                      f"最大: {np.max(self.fps_hist):.1f}")
            if self.err_hist:
                print(f"  误差: avg={np.mean(self.err_hist):.1f}cm  "
                      f"max={np.max(self.err_hist):.1f}cm  "
                      f"min={np.min(self.err_hist):.1f}cm")
        print(f"  {'─'*40}")

    def close(self):
        """清理资源。"""
        self.print_summary()
        if self.server_sock:
            self.server_sock.close()
        if self.client_sock:
            self.client_sock.close()
        if self.log_file:
            self.log_file.close()
            print(f"\n日志已保存: {self.log_file.name}")


# ==============================================================
# 入口
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="PC 端定位接收器")
    parser.add_argument("--port", type=int, default=9527, help="TCP 监听端口 (默认 9527)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")
    parser.add_argument("--no-log", action="store_true", help="不记录日志")
    args = parser.parse_args()

    receiver = PcReceiver(host=args.host, port=args.port, log=not args.no_log)

    try:
        receiver.run()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
