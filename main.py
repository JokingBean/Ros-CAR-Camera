#!/usr/bin/env python3
"""
ROS-Camera 三相机 BEV 系统 — 主菜单
====================================
python main.py
"""

import sys, os, yaml, time, socket
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PI_HOST = "100.126.101.5"
PI_USER = "pi"
PI_PASS = "alcht0"


def get_local_ip():
    """获取本机局域网 IP，优先 Tailscale。"""
    import subprocess
    # 优先 Tailscale IP (Pi 和 PC 通过 Tailscale 互通)
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5)
        ts_ip = result.stdout.strip()
        if ts_ip and "." in ts_ip:
            return ts_ip
    except Exception:
        pass
    # 回退：通过默认路由检测
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "?.?.?.?"


def load_config():
    """加载配置。"""
    with open(os.path.join(ROOT, "cfg", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def start_pi_tracker(pc_ip, port=9527):
    """SSH 到树莓派启动追踪服务（后台运行）。"""
    import paramiko
    print(f"\n  启动 Pi 追踪服务 ({PI_HOST})...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

        # 后台启动 pi_tracker.py，输出重定向到日志文件
        cmd = (f"cd /home/pi/uwb_tracker && "
               f"nohup python3 pi_tracker.py --pc-ip {pc_ip} --port {port} "
               f"> /tmp/pi_tracker.log 2>&1 &")
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=5)
        ssh.close()
        print(f"  Pi 追踪服务已在后台启动")
        print(f"  日志: /tmp/pi_tracker.log")
        print(f"  SSH 查看: ssh {PI_USER}@{PI_HOST} 'tail -f /tmp/pi_tracker.log'")
        return True
    except Exception as e:
        print(f"  SSH 启动失败: {e}")
        print(f"  请手动在 Pi 上运行:")
        print(f"    ssh {PI_USER}@{PI_HOST}")
        print(f"    cd /home/pi/uwb_tracker")
        print(f"    python3 pi_tracker.py --pc-ip {pc_ip} --port {port}")
        return False


def stop_pi_tracker():
    """停止 Pi 上的追踪服务。"""
    import paramiko
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
        ssh.exec_command("pkill -f pi_tracker.py 2>/dev/null", timeout=5)
        ssh.close()
        print("  Pi 追踪服务已停止")
        return True
    except Exception:
        return False


def menu():
    while True:
        print()
        print("=" * 55)
        print("  ROS-Camera 三相机 BEV 系统")
        print(f"  Pi: {PI_HOST}  |  PC: {get_local_ip()}")
        print("=" * 55)
        print("  1. 一键启动 Pi 追踪 + Web 控制台")
        print("  2. 精度测试（逐点定位 + 误差）")
        print("  3. 退出")
        print()

        choice = input("  > ").strip()
        if choice == "1":
            pc_ip = get_local_ip()
            started = start_pi_tracker(pc_ip, 9527)
            if started:
                time.sleep(2)
            try:
                from drivers.live_web import main as web_main
                web_main()
            except KeyboardInterrupt:
                pass
            # web_main 内部已处理 Ctrl+C 和 Pi 停止，直接退出
            print("  bye")
            break
        elif choice == "2":
            from drivers.precision import main
            main()
        elif choice == "3":
            print("  bye")
            break
        else:
            print("  无效选项")


if __name__ == "__main__":
    menu()
