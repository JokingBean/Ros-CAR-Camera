#!/usr/bin/env python3
"""
一键测试脚本 — 三相机小车追踪
=============================
直接调用 paramiko + OpenCV 抓图，不走嵌套脚本。
"""

import os, sys, time
from datetime import datetime
from pathlib import Path

PI_HOST = "100.101.225.34"
PI_USER = "pi"
PI_PASS = "alcht0"


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def main():
    # ================================================================
    step("1/3  抓取树莓派图像 (PiCamera + USB1)")
    # ================================================================
    try:
        import paramiko
    except ImportError:
        print("[!] 请先安装 paramiko: pip install paramiko")
        sys.exit(1)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    except Exception as e:
        print(f"[!] 树莓派连接失败: {e}")
        sys.exit(1)

    # 写拍摄脚本到 Pi
    capture_py = r"""#!/usr/bin/env python3
import cv2, time
from picamera2 import Picamera2
# 1) PiCamera
picam = Picamera2(0)
picam.configure(picam.create_video_configuration(main={'size': (1332, 990), 'format': 'RGB888'}, buffer_count=2))
picam.start(); time.sleep(1.0)
cv2.imwrite('/tmp/picam_cart.jpg', picam.capture_array())
picam.close(); time.sleep(0.3)
# 2) USB1
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
time.sleep(0.8)
for _ in range(8): cap.read()
ret, frame = cap.read()
if ret: cv2.imwrite('/tmp/usb1_cart.jpg', frame)
cap.release()
print('DONE')
"""

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/capture_pi.py", "w") as f:
        f.write(capture_py)
    sftp.close()

    stdin, stdout, stderr = ssh.exec_command("python3 /tmp/capture_pi.py", timeout=25)
    out = stdout.read().decode()
    err = stderr.read().decode().strip()
    if "DONE" not in out:
        print(f"[!] Pi 拍摄失败")
        print(f"    stdout: {out[:200]}")
        if err:
            for line in err.split("\n"):
                if "ERROR" in line or "Traceback" in line or "Error" in line:
                    print(f"    stderr: {line[:200]}")
        ssh.close()
        sys.exit(1)

    # 下载
    sftp = ssh.open_sftp()
    for f in ["picam_cart.jpg", "usb1_cart.jpg"]:
        try:
            sftp.get(f"/tmp/{f}", f)
            size = os.path.getsize(f)
            print(f"  {f}: {size//1024} KB")
        except Exception as e:
            print(f"[!] 下载 {f} 失败: {e}")
            ssh.close()
            sys.exit(1)
    sftp.close()
    ssh.close()
    print("  树莓派图像获取完成")

    # ================================================================
    step("2/3  抓取 USB2 图像 (本机)")
    # ================================================================
    import cv2
    usb2_ok = False
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(1.0)
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret and frame.mean() > 10:
            cv2.imwrite("usb2_cart.jpg", frame)
            print(f"  usb2_cart.jpg: {frame.shape[1]}x{frame.shape[0]} (idx={idx})")
            usb2_ok = True
            break
        else:
            m = frame.mean() if ret else -1
            print(f"  idx={idx}: mean={m:.0f} (dark/fail)")

    if not usb2_ok:
        print("[!] USB2 拍摄失败，检查相机连接")
        sys.exit(1)

    # ================================================================
    step("3/3  运行检测 + 融合 + 生成报告")
    # ================================================================
    import subprocess
    result = subprocess.run(
        [sys.executable, "cart_report.py"],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)

    if result.returncode != 0:
        print(f"[!] 报告生成失败:\n{result.stderr[:500]}")
        sys.exit(1)

    # 找结果文件夹
    dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
    if dirs:
        out_dir = dirs[0]
        report = out_dir / "cart_tracking_report.html"
        print(f"\n{'='*50}")
        print(f"  [OK] 完成!")
        print(f"  报告: {report}")
        print(f"  文件夹: {out_dir}")
        print(f"{'='*50}")

        # 自动打开报告
        import webbrowser
        webbrowser.open(str(report))
    else:
        print("[!] 未找到输出文件夹")


if __name__ == "__main__":
    main()
