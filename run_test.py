#!/usr/bin/env python3
"""
一键测试脚本 — 三相机小车追踪
=============================
自动从树莓派获取 PiCamera + USB1 图像，本地获取 USB2，
运行检测 + 融合，生成报告。

用法:
    python run_test.py

前置:
    - 树莓派已开机联网 (100.101.225.34)
    - 本机已连接 USB2 相机
    - pip install paramiko
"""

import subprocess, sys, os, time

# ==============================================================
PI_HOST = "100.101.225.34"
PI_USER = "pi"
PI_PASS = "alcht0"

# ==============================================================
def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")

def run(cmd, shell=True):
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.returncode != 0 and "WARN" not in result.stderr:
        # OpenCV warnings are OK
        real_err = [l for l in result.stderr.split('\n') if 'WARN' not in l and 'INFO' not in l]
        if real_err:
            print(f"  [!] {''.join(real_err)[:200]}")
    return result

# ==============================================================
def main():
    step("1/4  从树莓派获取 PiCamera + USB1 图像")

    capture_script = '''#!/usr/bin/env python3
import cv2, time
from picamera2 import Picamera2
picam = Picamera2(0)
picam.configure(picam.create_video_configuration(main={'size': (1332, 990), 'format': 'RGB888'}, buffer_count=2))
picam.start(); time.sleep(1.0)
cv2.imwrite('/tmp/picam_cart.jpg', picam.capture_array())
picam.close(); time.sleep(0.3)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
time.sleep(0.8)
for _ in range(8): cap.read()
ret, frame = cap.read()
if ret: cv2.imwrite('/tmp/usb1_cart.jpg', frame)
cap.release()
'''

    pi_code = f"""
import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect('{PI_HOST}', username='{PI_USER}', password='{PI_PASS}', timeout=10)
except Exception as e:
    print(f'SSH FAILED: {{e}}')
    print('Check Pi is online at {PI_HOST}')
    exit(1)

sftp = ssh.open_sftp()
with sftp.file('/tmp/cap.py', 'w') as f: f.write({repr(capture_script)})
sftp.close()
ssh.exec_command('python3 /tmp/cap.py 2>&1', timeout=25)
time.sleep(3)
sftp = ssh.open_sftp()
sftp.get('/tmp/picam_cart.jpg', 'picam_cart.jpg')
sftp.get('/tmp/usb1_cart.jpg', 'usb1_cart.jpg')
sftp.close()
ssh.close()
print('PiCamera + USB1 OK')
"""

    result = run(f'python -c "{pi_code}"')
    if result.returncode != 0:
        print("[FAIL] 树莓派连接失败，检查网络和IP")
        sys.exit(1)

    # Check files
    if not os.path.exists("picam_cart.jpg") or not os.path.exists("usb1_cart.jpg"):
        print("[FAIL] 未获取到树莓派图像")
        sys.exit(1)

    step("2/4  获取 USB2 图像（本机）")

    usb2_code = """
import cv2, time
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
time.sleep(1)
for _ in range(10): cap.read()
ret, frame = cap.read()
if ret:
    cv2.imwrite('usb2_cart.jpg', frame)
    print(f'USB2: {frame.shape[1]}x{frame.shape[0]} OK')
else:
    # try idx=0
    cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    time.sleep(1)
    for _ in range(10): cap2.read()
    ret2, frame2 = cap2.read()
    if ret2:
        cv2.imwrite('usb2_cart.jpg', frame2)
        print(f'USB2 (idx=0): {frame2.shape[1]}x{frame2.shape[0]} OK')
    else:
        print('FAIL: USB2 not found at idx 0 or 1')
    cap2.release()
cap.release()
"""
    result = run(f'python -c "{usb2_code}"')
    if not os.path.exists("usb2_cart.jpg"):
        print("[FAIL] USB2 图像获取失败，检查相机连接")
        sys.exit(1)

    step("3/4  运行检测 + 融合 + 生成报告")

    result = run("python cart_report.py")
    print(result.stdout)

    # 找最新生成的文件夹
    dirs = sorted([d for d in os.listdir() if d.startswith("tracking_run_")], reverse=True)
    if not dirs:
        print("[FAIL] 报告未生成")
        sys.exit(1)

    out_dir = dirs[0]

    step(f"4/4  完成 — {out_dir}")

    print(f"""
  ✅ PiCamera  图像已获取
  ✅ USB1      图像已获取
  ✅ USB2      图像已获取
  ✅ 检测 + 融合完成
  ✅ 报告: {out_dir}/cart_tracking_report.html

  用浏览器打开报告查看结果。
""")

if __name__ == "__main__":
    main()
