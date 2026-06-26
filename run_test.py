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
    step("3/4  自动外参标定 (请确保小车移出视野!)")

    import cv2, yaml as _y, numpy as np
    from pupil_apriltags import Detector

    with open("floor_tags.yaml", "r", encoding="utf-8") as f:
        _ft = _y.safe_load(f)
    floor_tags = {int(k): np.array([v['x'],v['y'],v['z']], dtype=np.float64) for k,v in _ft['tags'].items()}
    CART_TAGS = {0,1,2,3}; HALF_TAG = 0.045

    with open("config.yaml","r",encoding="utf-8") as f:
        _cfg = _y.safe_load(f)
    with open("extrinsics.yaml","r") as f:
        _ext = _y.safe_load(f)

    def auto_calib(name, img, scale, key, K, dist):
        if img is None: return f"{name}: 无图像"
        ih, iw = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if scale != 1.0: gray = cv2.resize(gray, None, fx=scale, fy=scale)
        gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray)
        if scale != 1.0:
            for d in dets: d.corners /= scale; d.center = (d.center[0]/scale, d.center[1]/scale)
        good = [d for d in dets if d.tag_id in floor_tags and d.tag_id not in CART_TAGS
                and 0.08*iw < d.center[0] < 0.92*iw and 0.08*ih < d.center[1] < 0.92*ih]
        if len(good) < 6: return f"{name}: Tag不足 ({len(good)})"
        obj_pts, img_pts = [], []
        for d in good:
            wpt = floor_tags[d.tag_id]
            c3 = np.array([[wpt[0]-HALF_TAG,wpt[1]-HALF_TAG,0],[wpt[0]+HALF_TAG,wpt[1]-HALF_TAG,0],
                          [wpt[0]+HALF_TAG,wpt[1]+HALF_TAG,0],[wpt[0]-HALF_TAG,wpt[1]+HALF_TAG,0]], dtype=np.float64)
            for ci, ii in zip(c3, d.corners): obj_pts.append(ci); img_pts.append(ii)
        obj_pts=np.array(obj_pts,dtype=np.float64); img_pts=np.array(img_pts,dtype=np.float64)
        ok, rv, tv, inl = cv2.solvePnPRansac(obj_pts, img_pts, K, dist, reprojectionError=4.0, confidence=0.99, iterationsCount=2000)
        if not ok: return f"{name}: PnP失败"
        R,_=cv2.Rodrigues(rv); pos = (-R.T@tv).flatten()
        n_in = len(inl) if inl is not None else 0
        errs = [np.linalg.norm(cv2.projectPoints(obj_pts[i].reshape(3,1),rv,tv,K,dist)[0].flatten()-img_pts[i]) for i in range(len(obj_pts))]
        _ext[key] = {'R': R.tolist(), 't': tv.flatten().tolist()}
        return f"{name}: {len(good)}tags err={np.mean(errs):.1f}px inliers={n_in}/{len(obj_pts)} H={abs(pos[2])*100:.0f}cm"

    # PiCamera
    picam_img = cv2.imread("picam_cart.jpg")
    cc = _cfg['cameras'][0]; Kp = np.array([[cc['camera_matrix']['fx'],0,cc['camera_matrix']['cx']],[0,cc['camera_matrix']['fy'],cc['camera_matrix']['cy']],[0,0,1]],dtype=np.float64)
    print(f"  {auto_calib('PiCam', picam_img, 1.0, 'picam_1', Kp, np.array(cc['dist_coeffs'],dtype=np.float64))}")

    # USB1
    usb1_img = cv2.imread("usb1_cart.jpg")
    cc = _cfg['cameras'][1]; Ku1 = np.array([[cc['camera_matrix']['fx'],0,cc['camera_matrix']['cx']],[0,cc['camera_matrix']['fy'],cc['camera_matrix']['cy']],[0,0,1]],dtype=np.float64)
    print(f"  {auto_calib('USB1', usb1_img, 1.0, 'usb_cam_1', Ku1, np.array(cc['dist_coeffs'],dtype=np.float64))}")

    # USB2
    usb2_img = cv2.imread("usb2_cart.jpg")
    cc = _cfg['cameras'][2]; Ku2 = np.array([[cc['camera_matrix']['fx'],0,cc['camera_matrix']['cx']],[0,cc['camera_matrix']['fy'],cc['camera_matrix']['cy']],[0,0,1]],dtype=np.float64)
    print(f"  {auto_calib('USB2', usb2_img, 0.5, 'usb_cam_2', Ku2, np.array(cc['dist_coeffs'],dtype=np.float64))}")

    with open("extrinsics.yaml","w") as f: _y.dump(_ext, f, default_flow_style=None)
    print("  外参已更新")

    # 标定完重新拍一次（因为小车可能已进入视野，刚才的图没小车Tag）
    print("  重新拍摄（小车进入视野）...")
    # Pi + USB1
    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    sftp = ssh.open_sftp()
    with sftp.file("/tmp/cap_cart.py","w") as f: f.write(capture_py)
    sftp.close(); ssh.exec_command("python3 /tmp/cap_cart.py", timeout=25); time.sleep(2)
    sftp = ssh.open_sftp()
    sftp.get("/tmp/picam_cart.jpg","picam_cart.jpg"); sftp.get("/tmp/usb1_cart.jpg","usb1_cart.jpg")
    sftp.close(); ssh.close()
    # USB2
    for idx in [1,0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(1.0)
        for _ in range(10): cap.read()
        ret, frame = cap.read(); cap.release()
        if ret and frame.mean() > 10: cv2.imwrite("usb2_cart.jpg", frame); break
    print("  重拍完成")

    # ================================================================
    step("4/4  运行检测 + 融合 + 生成报告")
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
