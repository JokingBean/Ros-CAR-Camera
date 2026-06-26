#!/usr/bin/env python3
"""ROS-Camera GUI v2 — 三相机小车追踪控制台"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading, queue, time, os, sys, subprocess
from datetime import datetime
from pathlib import Path
import cv2, yaml, numpy as np
from PIL import Image, ImageTk

APP_TITLE = "ROS-Camera 三相机小车追踪控制台"
PI_HOST = "100.101.225.34"
PI_USER = "pi"
PI_PASS = "alcht0"

class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1280x800")
        root.configure(bg="#1a1a2e")
        self.bev_image = None
        self.tracking_running = False
        self.camera_status = {}
        self.log_queue = queue.Queue()
        self._build_ui()
        self._process_log_queue()
        self.root.after(500, self.refresh_cameras)

    def _build_ui(self):
        bar = tk.Frame(self.root, bg="#0f3460", height=40)
        bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(bar, text=APP_TITLE, bg="#0f3460", fg="white", font=("Microsoft YaHei",14,"bold")).pack(side=tk.LEFT, padx=16, pady=5)
        tk.Label(bar, text="v2.0", bg="#0f3460", fg="#888", font=("Segoe UI",10)).pack(side=tk.RIGHT, padx=16, pady=5)

        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._build_camera_panel(main)
        self._build_bev_panel(main)
        self._build_control_panel(main)
        self._build_log_panel()

    def _build_camera_panel(self, parent):
        f = tk.LabelFrame(parent, text=" 摄像头 ", bg="#16213e", fg="#e0e0e0", font=("Microsoft YaHei",10), width=220)
        f.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4)); f.pack_propagate(False)
        bf = tk.Frame(f, bg="#16213e"); bf.pack(fill=tk.X, padx=6, pady=6)
        self.refresh_btn = tk.Button(bf, text="刷新检测", command=self.refresh_cameras, bg="#0f3460", fg="white", font=("Microsoft YaHei",9), relief=tk.FLAT, padx=12, pady=4)
        self.refresh_btn.pack(fill=tk.X)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)
        tk.Label(f, text="树莓派 (100.101.225.34)", bg="#16213e", fg="#e17055", font=("Microsoft YaHei",9,"bold")).pack(anchor=tk.W, padx=8, pady=(8,2))
        self.pi_frame = tk.Frame(f, bg="#16213e"); self.pi_frame.pack(fill=tk.X, padx=6)
        tk.Label(f, text="本机 PC", bg="#16213e", fg="#55efc4", font=("Microsoft YaHei",9,"bold")).pack(anchor=tk.W, padx=8, pady=(12,2))
        self.local_frame = tk.Frame(f, bg="#16213e"); self.local_frame.pack(fill=tk.X, padx=6)
        self.cam_status_label = tk.Label(f, text="等待检测...", bg="#16213e", fg="#888", font=("Microsoft YaHei",8))
        self.cam_status_label.pack(padx=8, pady=10)

    def _build_bev_panel(self, parent):
        f = tk.LabelFrame(parent, text=" 融合俯视图 (BEV) ", bg="#16213e", fg="#e0e0e0", font=("Microsoft YaHei",10))
        f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.bev_canvas = tk.Canvas(f, bg="#0a0a15", highlightthickness=0)
        self.bev_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.bev_text_id = self.bev_canvas.create_text(400,350, text="点击「场地融合」生成俯视图", fill="#444", font=("Microsoft YaHei",16))

    def _build_control_panel(self, parent):
        f = tk.LabelFrame(parent, text=" 控制 ", bg="#16213e", fg="#e0e0e0", font=("Microsoft YaHei",10), width=200)
        f.pack(side=tk.RIGHT, fill=tk.Y, padx=(4,0)); f.pack_propagate(False)
        bc = {"font":("Microsoft YaHei",9), "relief":tk.FLAT, "padx":10, "pady":6}
        tk.Label(f, text="标定", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=(10,4))
        tk.Button(f, text="内参标定", bg="#533a3a", fg="white", command=self.calibrate_intrinsic, **bc).pack(fill=tk.X, padx=6, pady=2)
        tk.Button(f, text="外参标定", bg="#533a3a", fg="white", command=self.calibrate_extrinsic, **bc).pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="融合", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        tk.Button(f, text="场地融合 (BEV)", bg="#2d5a2d", fg="white", command=self.do_fusion, **bc).pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="追踪", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        tk.Button(f, text="小车定位 (单次)", bg="#2d2d5a", fg="white", command=self.do_tracking_once, **bc).pack(fill=tk.X, padx=6, pady=2)
        self.btn_live = tk.Button(f, text="实时追踪 (开始)", bg="#2d2d5a", fg="white", command=self.toggle_live_tracking, **bc)
        self.btn_live.pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="融合模式", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        self.fusion_mode = tk.StringVar(value="gsd_weighted")
        for t,v in [("GSD 加权","gsd_weighted"),("最佳选择","best_select"),("简单平均","average")]:
            tk.Radiobutton(f, text=t, variable=self.fusion_mode, value=v, bg="#16213e", fg="#ccc", selectcolor="#0f3460", font=("Microsoft YaHei",8)).pack(anchor=tk.W, padx=16)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Button(f, text="导出完整报告", bg="#4a3a2d", fg="white", command=self.export_report, **bc).pack(fill=tk.X, padx=6, pady=2)
        tk.Label(f, text="● PiCam  ● USB1  ● USB2  ● 小车", bg="#16213e", fg="#888", font=("Microsoft YaHei",7)).pack(pady=10)

    def _build_log_panel(self):
        self.log_text = scrolledtext.ScrolledText(self.root, height=6, bg="#0a0a15", fg="#aaa", font=("Consolas",9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0,4))
        self.log_text.insert(tk.END, "ROS-Camera 控制台就绪\n")

    def log(self, msg):
        for line in msg.split('\n'):
            self.log_queue.put(line)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_text.see(tk.END)
        self.root.after(100, self._process_log_queue)

    # ==================================================================
    def refresh_cameras(self):
        self.log("正在检测摄像头...")
        self.cam_status_label.config(text="检测中...", fg="#e17055")
        def _detect():
            status = {}
            for idx in [1,2]:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                    ret,_=cap.read(); cap.release()
                    if ret: status[f"local_{idx}"] = True
            try:
                import paramiko
                ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=5)
                stdin,stdout,stderr=ssh.exec_command('ls /dev/video2 2>/dev/null&&echo PICAM_OK;ls /dev/video0 2>/dev/null&&echo USB_OK')
                out=stdout.read().decode()
                status["pi_picam"]="PICAM_OK" in out; status["pi_usb"]="USB_OK" in out
                status["pi_online"]=True; ssh.close()
            except: status["pi_online"]=False
            self.camera_status=status
            self.root.after(0,self._update_camera_list,status)
        threading.Thread(target=_detect, daemon=True).start()

    def _update_camera_list(self, status):
        for w in self.pi_frame.winfo_children(): w.destroy()
        for w in self.local_frame.winfo_children(): w.destroy()
        pi_ok = status.get("pi_online",False)
        tk.Label(self.pi_frame, text="在线" if pi_ok else "离线", bg="#16213e", fg="#55efc4" if pi_ok else "#e17055", font=("Microsoft YaHei",8)).pack(anchor=tk.W, padx=2)
        for k,l in [("pi_picam","PiCamera"),("pi_usb","USB1")]:
            ok = status.get(k,False)
            tk.Label(self.pi_frame, text=f"  {'●' if ok else '○'} {l}", bg="#16213e", fg="#55efc4" if ok else "#e17055", font=("Microsoft YaHei",9)).pack(anchor=tk.W, padx=10)
        tk.Label(self.local_frame, text=f"发现 {sum(1 for k,v in status.items() if k.startswith('local_') and v)} 个", bg="#16213e", fg="#888", font=("Microsoft YaHei",8)).pack(anchor=tk.W, padx=2)
        for i in [1,2]:
            ok = status.get(f"local_{i}",False)
            tk.Label(self.local_frame, text=f"  {'●' if ok else '○'} {'USB2' if i==1 else f'idx={i}'}", bg="#16213e", fg="#55efc4" if ok else "#555", font=("Microsoft YaHei",9)).pack(anchor=tk.W, padx=10)
        self.cam_status_label.config(text="检测完成", fg="#888")
        self.log("摄像头检测完成")

    # ==================================================================
    def _run_in_thread(self, target, on_done=None):
        def w():
            try:
                r = target()
                if on_done: self.root.after(0, lambda: on_done(r))
            except Exception as e:
                self.log(f"错误: {e}")
                import traceback; traceback.print_exc()
        threading.Thread(target=w, daemon=True).start()

    def _choose_camera(self, title):
        dlg = tk.Toplevel(self.root); dlg.title(title); dlg.geometry("320x150")
        dlg.configure(bg="#16213e"); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg, text="选择相机:", bg="#16213e", fg="white", font=("Microsoft YaHei",10)).pack(pady=(16,8))
        choices = ["picam (PiCamera)","usb (USB1)","usb2 (USB2)"]
        vals = ["picam","usb","usb2"]
        var = tk.StringVar(value=choices[2])
        cb = ttk.Combobox(dlg, textvariable=var, values=choices, state="readonly", font=("Microsoft YaHei",10), width=28)
        cb.pack(padx=16, pady=4); cb.current(2)
        result = [None]
        def ok(): result[0]=vals[cb.current()]; dlg.destroy()
        bf = tk.Frame(dlg, bg="#16213e"); bf.pack(pady=10)
        tk.Button(bf, text="确定", command=ok, bg="#0f3460", fg="white", relief=tk.FLAT, padx=16).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="取消", command=dlg.destroy, bg="#333", fg="white", relief=tk.FLAT, padx=16).pack(side=tk.LEFT, padx=4)
        self.root.wait_window(dlg)
        return result[0]

    def calibrate_intrinsic(self):
        c = self._choose_camera("内参标定 — 选择相机")
        if not c: return
        self.log(f"启动内参标定: {c} (在新窗口中，按 's' 保存, 'q' 结束)")
        def t():
            r = subprocess.run(
                [sys.executable, "calibrate_intrinsics.py", "--camera", c],
                cwd="calibration_toolkit",
                capture_output=False,  # 不捕获，让 OpenCV 窗口弹出
                timeout=600)
            # 读结果文件
            result_file = Path(f"calibration_toolkit/camera_calibration_{c}.json")
            if result_file.exists():
                import json
                with open(result_file) as f: data = json.load(f)
                fx = data['camera_matrix']['fx']
                fy = data['camera_matrix']['fy']
                err = data['reprojection_error']
                n = data['num_images']
                return f"内参标定完成 ({c}): fx={fx:.1f} fy={fy:.1f} 误差={err:.3f}px ({n}张)"
            return f"内参完成 ({c})"
        self._run_in_thread(t, lambda m: self.log(m))

    def calibrate_extrinsic(self):
        c = self._choose_camera("外参标定 — 选择相机")
        if not c: return
        self.log(f"自动外参标定: {c} (请确保小车移出视野)")
        def t():
            import yaml as y, time as tm
            with open("floor_tags.yaml","r",encoding="utf-8") as f: ft=y.safe_load(f)
            floor_tags={int(k):np.array([v['x'],v['y'],v['z']],dtype=np.float64) for k,v in ft['tags'].items()}
            CART_IDS={0,1,2,3}
            cam_key={"picam":"picam_1","usb":"usb_cam_1","usb2":"usb_cam_2"}[c]
            with open("config.yaml","r",encoding="utf-8") as f: cfg=y.safe_load(f)
            cc=next(cam for cam in cfg['cameras'] if cam['name']==cam_key)
            cm=cc['camera_matrix']; K=np.array([[cm['fx'],0,cm['cx']],[0,cm['fy'],cm['cy']],[0,0,1]],dtype=np.float64)
            dist=np.array(cc['dist_coeffs'],dtype=np.float64)
            # 拍图
            if c=="picam" or c=="usb":
                import paramiko
                ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST,username=PI_USER,password=PI_PASS,timeout=10)
                if c=="picam":
                    scr='''#!/usr/bin/env python3\nimport cv2,time\nfrom picamera2 import Picamera2\npicam=Picamera2(0)\npicam.configure(picam.create_video_configuration(main={'size':(1332,990),'format':'RGB888'},buffer_count=2))\npicam.start();time.sleep(1.0)\ncv2.imwrite('/tmp/ext.jpg',picam.capture_array())\npicam.close()\n'''
                else:
                    scr='''#!/usr/bin/env python3\nimport cv2,time\ncap=cv2.VideoCapture(0,cv2.CAP_V4L2)\ncap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))\ncap.set(cv2.CAP_PROP_FRAME_WIDTH,2048);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1536)\ntime.sleep(1.0)\nfor _ in range(10):cap.read()\nret,frame=cap.read()\nif ret:cv2.imwrite('/tmp/ext.jpg',frame)\ncap.release()\n'''
                sftp=ssh.open_sftp()
                with sftp.file("/tmp/cap_e.py","w") as f:f.write(scr)
                sftp.close();ssh.exec_command("python3 /tmp/cap_e.py",timeout=20);tm.sleep(2)
                sftp=ssh.open_sftp();sftp.get("/tmp/ext.jpg","_ext.jpg");sftp.close();ssh.close()
                frame=cv2.imread("_ext.jpg")
                if frame is None: return f"{c} 拍摄失败"
                ih,iw=frame.shape[:2]; scale=1.0
            else:
                for idx in[1,0]:
                    cap=cv2.VideoCapture(idx,cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1440)
                    tm.sleep(1.0)
                    for _ in range(10):cap.read()
                    ret,frame=cap.read();cap.release()
                    if ret and frame.mean()>10:break
                if frame is None or frame.mean()<10:return "USB2 拍摄失败"
                ih,iw=frame.shape[:2]; scale=0.5
            # 检测
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if scale!=1.0:gray=cv2.resize(gray,None,fx=scale,fy=scale)
            gray=cv2.createCLAHE(2.0,(8,8)).apply(gray)
            from pupil_apriltags import Detector
            dets=Detector(families="tag36h11",quad_decimate=1.0).detect(gray)
            if scale!=1.0:
                for d in dets:d.corners/=scale;d.center=(d.center[0]/scale,d.center[1]/scale)
            floor_dets=[d for d in dets if d.tag_id in floor_tags and d.tag_id not in CART_IDS]
            if len(floor_dets)<6:return f"Tag不足 ({len(floor_dets)})"
            # 过滤边缘
            m=0.08
            good=[d for d in floor_dets if m*iw<d.center[0]<(1-m)*iw and m*ih<d.center[1]<(1-m)*ih]
            # PnP
            half=0.045; obj_pts,img_pts=[],[]
            for d in good:
                wpt=floor_tags[d.tag_id]
                c3=np.array([[wpt[0]-half,wpt[1]-half,0],[wpt[0]+half,wpt[1]-half,0],[wpt[0]+half,wpt[1]+half,0],[wpt[0]-half,wpt[1]+half,0]],dtype=np.float64)
                for ci,ii in zip(c3,d.corners):obj_pts.append(ci);img_pts.append(ii)
            obj_pts=np.array(obj_pts,dtype=np.float64);img_pts=np.array(img_pts,dtype=np.float64)
            ok,rv,tv,inl=cv2.solvePnPRansac(obj_pts,img_pts,K,dist,reprojectionError=4.0,confidence=0.99,iterationsCount=2000)
            if not ok:return "PnP失败"
            R,_=cv2.Rodrigues(rv);pos=(-R.T@tv).flatten()
            n_in=len(inl) if inl is not None else 0
            errs=[np.linalg.norm(cv2.projectPoints(obj_pts[i].reshape(3,1),rv,tv,K,dist)[0].flatten()-img_pts[i]) for i in range(len(obj_pts))]
            with open("extrinsics.yaml","r") as f:ext_all=y.safe_load(f)
            ext_all[cam_key]={'R':R.tolist(),'t':tv.flatten().tolist()}
            with open("extrinsics.yaml","w") as f:y.dump(ext_all,f,default_flow_style=None)
            return f"外参标定完成 ({c}): {len(good)}tags, inliers={n_in}/{len(obj_pts)}, err={np.mean(errs):.1f}px, H={abs(pos[2])*100:.0f}cm, pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
        self._run_in_thread(t, lambda m: self.log(m))

    def do_fusion(self):
        self.log("正在拍摄三相机图像...")

        def task():
            import paramiko, time as tm
            # 1) 树莓派拍图
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
                sftp = ssh.open_sftp()
                with sftp.file("/tmp/cap_pi.py", "w") as f:
                    f.write(r"""#!/usr/bin/env python3
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
print('DONE')
""")
                sftp.close()
                stdin, stdout, stderr = ssh.exec_command("python3 /tmp/cap_pi.py", timeout=25)
                if b"DONE" not in stdout.read():
                    ssh.close(); return "树莓派拍摄失败"
                tm.sleep(1)
                sftp = ssh.open_sftp()
                sftp.get("/tmp/picam_cart.jpg", "picam_cart.jpg")
                sftp.get("/tmp/usb1_cart.jpg", "usb1_cart.jpg")
                sftp.close(); ssh.close()
                self.log("PiCamera + USB1 已获取")
            except Exception as e:
                return f"树莓派连接失败: {e}"

            # 2) USB2 本机拍图
            usb_ok = False
            for idx in [1, 0]:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                tm.sleep(1.0)
                for _ in range(10): cap.read()
                ret, frame = cap.read()
                cap.release()
                if ret and frame.mean() > 10:
                    cv2.imwrite("usb2_cart.jpg", frame)
                    self.log(f"USB2 已获取 (idx={idx})")
                    usb_ok = True
                    break
            if not usb_ok:
                return "USB2 拍摄失败"

            # 3) 生成报告 + BEV
            r = subprocess.run([sys.executable, "cart_report.py"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return f"报告生成失败: {r.stderr[:200]}"
            dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
            if dirs and (dirs[0] / "cart_bev.jpg").exists():
                return str(dirs[0] / "cart_bev.jpg")
            return "BEV 生成失败"

        def on_done(result):
            if result and result.endswith(".jpg"):
                self._load_bev(result)
                self.log(f"BEV 已加载")
            else:
                self.log(f"场地融合失败: {result}")

        self._run_in_thread(task, on_done)

    def _load_bev(self, path):
        img = Image.open(path)
        cw = self.bev_canvas.winfo_width() or 700
        ch = self.bev_canvas.winfo_height() or 600
        img.thumbnail((cw,ch), Image.LANCZOS)
        self.bev_image = ImageTk.PhotoImage(img)
        self.bev_canvas.delete("all")
        self.bev_canvas.create_image(cw//2, ch//2, image=self.bev_image, anchor=tk.CENTER)

    def do_tracking_once(self):
        self.log("低延时定位中 (Pi端检测+JSON回传)...")
        def task():
            import json, time as tm, paramiko, yaml as _y
            t_start = tm.time()
            timings = {}

            # === Pi 端检测 ===
            t0 = tm.time()
            try:
                ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
                # 上传 tracker（首次）
                sftp = ssh.open_sftp()
                try: sftp.stat("/home/pi/UwbCamera/pi_tracker.py")
                except:
                    with open("pi_tracker.py","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/pi_tracker.py")
                    with open("extrinsics.yaml","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/extrinsics.yaml")
                sftp.close()
                stdin, stdout, stderr = ssh.exec_command(
                    "cd /home/pi/UwbCamera && python3 pi_tracker.py 2>/dev/null", timeout=10)
                pi_out = stdout.read().decode().strip()
                pi_results = json.loads(pi_out) if pi_out else []
                ssh.close()
            except Exception as e:
                pi_results = []
                self.log(f"Pi 检测失败: {e}")
            timings["Pi检测"] = (tm.time()-t0)*1000

            # === USB2 检测 ===
            t0 = tm.time()
            usb2_results = []
            for idx in [1,0]:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                tm.sleep(0.2)
                for _ in range(5): cap.read()
                ret, frame = cap.read(); cap.release()
                if ret and frame.mean() > 10: break
            if frame is not None:
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
                gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
                from pupil_apriltags import Detector
                dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray)
                # USB2 solvePose
                with open("extrinsics.yaml","r") as f: ext = _y.safe_load(f)
                R2=np.array(ext["usb_cam_2"]["R"]); t2=np.array(ext["usb_cam_2"]["t"]).reshape(3,1)
                K2=np.array([[1997.5587,0,1203.9179],[0,2004.3731,784.2230],[0,0,1]],dtype=np.float64)
                D2=np.array([0.08367,-0.15649,0.00321,-0.00835,0.11271],dtype=np.float64)
                half=0.0675; obj=np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],dtype=np.float64)
                for d in dets:
                    if d.tag_id in {0,1,2,3}:
                        d.corners*=2.0; d.center=(d.center[0]*2,d.center[1]*2)
                        ok,rv,tv=cv2.solvePnP(obj,d.corners,K2,D2)
                        if ok:
                            Rt,_=cv2.Rodrigues(rv);tt=tv.reshape(3,1);Rc=R2.T;tc=-Rc@t2
                            tw=(Rc@tt+tc).flatten();gsd=np.linalg.norm(R2@tw.reshape(3,1)+t2)/((K2[0,0]+K2[1,1])/2)*1000
                            usb2_results.append({"tag_id":int(d.tag_id),"position":tw.tolist(),"gsd":round(float(gsd),2),"source":"USB2"})
            timings["USB2检测"] = (tm.time()-t0)*1000

            # === 融合 ===
            t0 = tm.time()
            all_r = pi_results + usb2_results
            fused = None
            if all_r:
                w = np.array([1.0/max(r["gsd"],0.01) for r in all_r]); w/=w.sum()
                pos = np.zeros(3)
                for wi, r in zip(w, all_r): pos += wi * np.array(r["position"])
                best = min(all_r, key=lambda r: r["gsd"])
                fused = {"position": pos, "gsd": best["gsd"],
                         "sources": list(set(r.get("source","?") for r in all_r))}
            timings["融合"] = (tm.time()-t0)*1000

            total = (tm.time()-t_start)*1000
            if fused:
                p=fused["position"]; srcs=",".join(fused["sources"])
                timings_str = " | ".join(f"{k}:{v:.0f}ms" for k,v in timings.items())
                return f"定位完成 ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) gsd={fused['gsd']:.1f}mm [{srcs}] | 总:{total:.0f}ms | {timings_str}"
            return f"未检测到小车 | {timings_str}"
        self._run_in_thread(task, lambda m: self.log(m))

    def toggle_live_tracking(self):
        if self.tracking_running:
            self.tracking_running = False
            self.btn_live.config(text="实时追踪 (开始)", bg="#2d2d5a")
            self.log("实时追踪已停止")
        else:
            self.tracking_running = True
            self.btn_live.config(text="实时追踪 (停止)", bg="#5a2d2d")
            self.log("实时追踪已启动 (每2秒)")
            self._live_loop()

    def _live_loop(self):
        if not self.tracking_running: return
        def t(): subprocess.run([sys.executable,"cart_report.py"], capture_output=True, timeout=120); return True
        def d(_):
            if self.tracking_running:
                dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
                if dirs and (dirs[0]/"cart_bev.jpg").exists(): self._load_bev(str(dirs[0]/"cart_bev.jpg"))
                self.root.after(2000, self._live_loop)
        self._run_in_thread(t, d)

    def export_report(self):
        dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
        if dirs and (dirs[0]/"cart_tracking_report.html").exists():
            import webbrowser; webbrowser.open(str(dirs[0]/"cart_tracking_report.html"))
            self.log("已打开报告")
        else: messagebox.showwarning("导出","没有找到报告，请先执行一次定位")

def main():
    root = tk.Tk(); App(root); root.mainloop()

if __name__ == "__main__": main()
