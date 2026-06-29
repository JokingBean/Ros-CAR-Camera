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
        root.geometry("1280x900")
        root.configure(bg="#1a1a2e")
        self.bev_image = None
        self.tracking_running = False
        self.camera_status = {}
        self.log_queue = queue.Queue()
        # USB2 常量化（避免每帧重复加载）
        self._usb2_init()
        import yaml
        with open("extrinsics.yaml","r") as f: self._ext = yaml.safe_load(f)
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
        # 标定按钮
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6)
        bc_small = {"font":("Microsoft YaHei",8), "relief":tk.FLAT, "padx":6, "pady":3}
        tk.Button(f, text="内参标定", bg="#533a3a", fg="white", command=self.calibrate_intrinsic, **bc_small).pack(fill=tk.X, padx=6, pady=1)
        tk.Button(f, text="外参标定", bg="#533a3a", fg="white", command=self.calibrate_extrinsic, **bc_small).pack(fill=tk.X, padx=6, pady=1)

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
        tk.Label(f, text="融合", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        tk.Button(f, text="场地融合 (BEV)", bg="#2d5a2d", fg="white", command=self.do_fusion, **bc).pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="追踪", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        tk.Button(f, text="小车定位 (单次)", bg="#2d2d5a", fg="white", command=self.do_tracking_once, **bc).pack(fill=tk.X, padx=6, pady=2)
        self.btn_live = tk.Button(f, text="实时追踪 (开始)", bg="#2d2d5a", fg="white", command=self.toggle_live_tracking, **bc)
        self.btn_live.pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="精度验证 (0.5m自动网格)", bg="#16213e", fg="#55efc4", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        pf = tk.Frame(f, bg="#16213e"); pf.pack(fill=tk.X, padx=6, pady=2)
        self.gt_label = tk.Label(pf, text="---", bg="#16213e", fg="#55efc4", font=("Microsoft YaHei",9,"bold"))
        self.gt_label.pack(side=tk.LEFT, padx=4)
        tk.Button(f, text="精度测量 (单次)", bg="#3a4a2d", fg="white", command=self.precision_measure, **bc).pack(fill=tk.X, padx=6, pady=2)
        tk.Button(f, text="生成精度报告", bg="#3a4a2d", fg="white", command=self.precision_report, **bc).pack(fill=tk.X, padx=6, pady=2)
        tk.Button(f, text="清除全部数据", bg="#5a3a3a", fg="white", command=self.clear_precision, **bc).pack(fill=tk.X, padx=6, pady=2)
        self.precision_count = tk.Label(f, text="已测: 0 次", bg="#16213e", fg="#888", font=("Microsoft YaHei",8))
        self.precision_count.pack(padx=6)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Label(f, text="融合模式", bg="#16213e", fg="#e94560", font=("Microsoft YaHei",10,"bold")).pack(anchor=tk.W, padx=8, pady=4)
        self.fusion_mode = tk.StringVar(value="gsd_weighted")
        for t,v in [("GSD 加权","gsd_weighted"),("最佳选择","best_select"),("简单平均","average")]:
            tk.Radiobutton(f, text=t, variable=self.fusion_mode, value=v, bg="#16213e", fg="#ccc", selectcolor="#0f3460", font=("Microsoft YaHei",8)).pack(anchor=tk.W, padx=16)
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=8)
        tk.Button(f, text="导出完整报告", bg="#4a3a2d", fg="white", command=self.export_report, **bc).pack(fill=tk.X, padx=6, pady=2)

        tk.Label(f, text="● PiCam  ● USB1  ● USB2", bg="#16213e", fg="#888", font=("Microsoft YaHei",7)).pack(pady=8)

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
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                        ret,_=cap.read(); cap.release()
                        if ret: status[f"local_{idx}"] = True
                except:
                    try: cap.release()
                    except: pass
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
                    scr='''#!/usr/bin/env python3\nimport cv2,time\nfrom picamera2 import Picamera2\npicam=Picamera2(0)\npicam.configure(picam.create_video_configuration(main={'size':(2028,1520),'format':'RGB888'},buffer_count=1))\npicam.start();time.sleep(0.3)\ncv2.imwrite('/tmp/ext.jpg',picam.capture_array())\npicam.close()\n'''
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
            floor_dets=[d for d in dets if d.tag_id in floor_tags]
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
                # 暂停 TCP 服务器
                ssh.exec_command("pkill -f pi_tracker_server.py; sleep 1")
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
                sftp.close()
                # 重启 TCP 服务器
                ssh.exec_command("cd /home/pi/UwbCamera && nohup python3 pi_tracker_server.py > /tmp/pi_tracker.log 2>&1 &")
                ssh.close()
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
        self.log("低延时定位中 (Pi持续服务)...")
        def task():
            import json, time as tm, socket, threading as _th, yaml as _y

            t_start = tm.time(); timings = {}
            pi_results = []; usb2_results = []; t_usb2_cap = 0

            # === Pi TCP + USB2 并行 ===
            def fetch_pi():
                nonlocal pi_results
                for attempt in range(2):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        sock.connect((PI_HOST, 9999))
                        data = b""
                        while b"\n" not in data:
                            chunk = sock.recv(4096)
                            if not chunk: break
                            data += chunk
                        sock.close()
                        msg = json.loads(data.decode().strip())
                        pi_results = msg.get("results", [])
                        timings["Pi耗时"] = msg.get("elapsed_ms", 0)
                        return
                    except:
                        if attempt == 0:
                            try:
                                import paramiko
                                ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
                                sftp = ssh.open_sftp()
                                with open("pi_tracker_server.py","rb") as sf: sftp.putfo(sf, "/home/pi/UwbCamera/pi_tracker_server.py")
                                with open("extrinsics.yaml","rb") as sf: sftp.putfo(sf, "/home/pi/UwbCamera/extrinsics.yaml")
                                sftp.close()
                                ssh.exec_command("pkill -f pi_tracker_server.py 2>/dev/null; sleep 1.5; cd /home/pi/UwbCamera && python3 pi_tracker_server.py > /tmp/pi_tracker.log 2>&1 &")
                                ssh.close()
                                tm.sleep(2.5)
                            except:
                                pass
                        else:
                            self.log("Pi TCP失败，请手动SSH运行 pi_tracker_server.py")

            def capture_usb2():
                nonlocal t_usb2_cap, usb2_results
                t_usb2_cap = tm.time()
                frame = None
                for idx in [1]:  # skip idx=0
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                    tm.sleep(0.1)
                    for _ in range(2): cap.read()
                    ret, frame = cap.read(); t_usb2_cap = tm.time(); cap.release()
                    if ret and frame.mean() > 10: break
                if frame is not None:
                    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
                    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
                    from pupil_apriltags import Detector
                    dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray)
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
                                usb2_results.append({"tag_id":int(d.tag_id),"position":tw.tolist(),"gsd":round(float(gsd),2),"source":"USB2","t_capture":round(t_usb2_cap,3)})

            t0 = tm.time()
            t_pi = _th.Thread(target=fetch_pi); t_usb = _th.Thread(target=capture_usb2)
            t_pi.start(); t_usb.start()
            t_pi.join(); t_usb.join()
            timings["并行总耗时"] = (tm.time()-t0)*1000

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
                pi_ts = [r["t_capture"] for r in pi_results if "t_capture" in r]
                usb_ts = [r["t_capture"] for r in usb2_results if "t_capture" in r]
                sync_info = ""
                if pi_ts and usb_ts:
                    delta = abs(np.mean(pi_ts) - np.mean(usb_ts)) * 1000
                    sync_info = f" | 同步偏差:{delta:.0f}ms"
                timings_str = " | ".join(f"{k}:{v:.0f}ms" for k,v in timings.items())
                return f"定位完成 ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) gsd={fused['gsd']:.1f}mm [{srcs}]{sync_info} | 总:{total:.0f}ms | {timings_str}"
            return f"未检测到小车 | " + " | ".join(f"{k}:{v:.0f}ms" for k,v in timings.items())
        self._run_in_thread(task, lambda m: self.log(m))
        def task():
            import json, time as tm, paramiko, yaml as _y
            t_start = tm.time()
            timings = {}

            # === Pi端 + USB2 并行抓图 ===
            t0 = tm.time()
            t_start = tm.time()
            timings = {}
            pi_results = []; usb2_results = []; t_usb2_cap = 0

            # 并行：SSH Pi + USB2 抓图
            def capture_pi():
                nonlocal pi_results
                try:
                    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
                    sftp = ssh.open_sftp()
                    try: sftp.stat("/home/pi/UwbCamera/pi_tracker.py")
                    except:
                        with open("pi_tracker.py","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/pi_tracker.py")
                        with open("extrinsics.yaml","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/extrinsics.yaml")
                    sftp.close()
                    stdin, stdout, stderr = ssh.exec_command(
                        "cd /home/pi/UwbCamera && python3 pi_tracker.py 2>/dev/null", timeout=10)
                    out = stdout.read().decode().strip()
                    pi_results = json.loads(out) if out else []
                    ssh.close()
                except Exception as e:
                    self.log(f"Pi检测失败: {e}")

            def capture_usb2():
                nonlocal t_usb2_cap, usb2_results
                t_usb2_cap = tm.time()
                frame = None
                for idx in [1]:  # skip idx=0
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                    tm.sleep(0.15)
                    for _ in range(3): cap.read()
                    ret, frame = cap.read(); t_usb2_cap = tm.time(); cap.release()
                    if ret and frame.mean() > 10: break
                if frame is not None:
                    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
                    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
                    from pupil_apriltags import Detector
                    dets = Detector(families="tag36h11", quad_decimate=1.0).detect(gray)
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
                                usb2_results.append({"tag_id":int(d.tag_id),"position":tw.tolist(),"gsd":round(float(gsd),2),"source":"USB2","t_capture":round(t_usb2_cap,3)})

            import threading as _th
            t_pi = _th.Thread(target=capture_pi); t_usb = _th.Thread(target=capture_usb2)
            t_pi.start(); t_usb.start()
            t_pi.join(); t_usb.join()
            timings["并行抓图+检测"] = (tm.time()-t0)*1000

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
                # 时间同步分析
                pi_ts = [r["t_capture"] for r in pi_results if "t_capture" in r]
                usb_ts = [r["t_capture"] for r in usb2_results if "t_capture" in r]
                sync_info = ""
                if pi_ts and usb_ts:
                    delta = abs(np.mean(pi_ts) - np.mean(usb_ts)) * 1000
                    sync_info = f" | 同步偏差:{delta:.0f}ms"
                timings_str = " | ".join(f"{k}:{v:.0f}ms" for k,v in timings.items())
                return f"定位完成 ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) gsd={fused['gsd']:.1f}mm [{srcs}]{sync_info} | 总:{total:.0f}ms | {timings_str}"
            return f"未检测到小车 | " + " | ".join(f"{k}:{v:.0f}ms" for k,v in timings.items())
        self._run_in_thread(task, lambda m: self.log(m))

    def toggle_live_tracking(self):
        if self.tracking_running:
            self.tracking_running = False
            self.btn_live.config(text="实时追踪 (开始)", bg="#2d2d5a")
            self.log("实时追踪已停止")
            if self._pi_sock:
                try: self._pi_sock.close()
                except: pass
                self._pi_sock = None
        else:
            self.tracking_running = True
            self.btn_live.config(text="实时追踪 (停止)", bg="#5a2d2d")
            self._fps_history = []
            self._pi_sock = None
            self._live_loop()

    def _usb2_init(self):
        """预计算 USB2 的固定矩阵，避免每帧重复加载。"""
        import yaml
        with open("extrinsics.yaml","r") as f: ext = yaml.safe_load(f)
        self._usb2_R = np.array(ext["usb_cam_2"]["R"])
        self._usb2_t = np.array(ext["usb_cam_2"]["t"]).reshape(3,1)
        self._usb2_K = np.array([[1997.5587,0,1203.9179],[0,2004.3731,784.2230],[0,0,1]], dtype=np.float64)
        self._usb2_D = np.array([0.08367,-0.15649,0.00321,-0.00835,0.11271], dtype=np.float64)
        self._usb2_Rc = self._usb2_R.T
        self._usb2_tc = -self._usb2_Rc @ self._usb2_t
        h = 0.0675
        self._usb2_obj = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], dtype=np.float64)

    def _start_pi_server(self):
        """SSH 到 Pi 启动 pi_tracker_server.py（异步）。"""
        import threading
        threading.Thread(target=self._start_pi_server_sync, daemon=True).start()

    def _start_pi_server_sync(self):
        """SSH 到 Pi 启动服务（阻塞直到完成）。"""
        import paramiko
        try:
            ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
            # 检查是否已运行
            stdin, stdout, stderr = ssh.exec_command("pgrep -f pi_tracker_server.py")
            pid = stdout.read().decode().strip()
            if pid:
                self.log("Pi 追踪服务已在运行")
            else:
                # 上传脚本
                sftp = ssh.open_sftp()
                with open("pi_tracker_server.py","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/pi_tracker_server.py")
                with open("extrinsics.yaml","rb") as f: sftp.putfo(f, "/home/pi/UwbCamera/extrinsics.yaml")
                sftp.close()
                # 后台启动
                ssh.exec_command("cd /home/pi/UwbCamera && nohup python3 pi_tracker_server.py > /tmp/pi_tracker.log 2>&1 &")
                import time as tm; tm.sleep(1)
                self.log("Pi 追踪服务已启动")
            ssh.close()
            self._pi_server_ready = True
        except Exception as e:
            self.log(f"Pi 服务启动失败: {e}")
            self._pi_server_ready = True  # 继续尝试

    def _live_loop(self):
        if not self.tracking_running: return
        import socket, time as tm, threading as _th, json
        t0 = tm.time()
        pi_results = []
        def fetch_pi():
            nonlocal pi_results
            try:
                if self._pi_sock is None:
                    self._pi_sock = socket.socket(); self._pi_sock.settimeout(5)
                    self._pi_sock.connect((PI_HOST, 9999))
                self._pi_sock.sendall(b"tick\n")
                data = b""
                while b"\n" not in data:
                    c = self._pi_sock.recv(4096)
                    if not c: break
                    data += c
                pi_results = json.loads(data.decode().strip()).get("results",[])
            except:
                self._pi_sock = None; pi_results = []
                # 首次失败尝试自动启动
                if not hasattr(self, "_pi_auto_started"):
                    self._pi_auto_started = True
                    try:
                        import paramiko
                        ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=8)
                        sftp = ssh.open_sftp()
                        with open("pi_tracker_server.py","rb") as sf: sftp.putfo(sf, "/home/pi/UwbCamera/pi_tracker_server.py")
                        with open("extrinsics.yaml","rb") as sf: sftp.putfo(sf, "/home/pi/UwbCamera/extrinsics.yaml")
                        sftp.close()
                        ssh.exec_command("pkill -f pi_tracker_server.py 2>/dev/null; sleep 1.5; cd /home/pi/UwbCamera && python3 pi_tracker_server.py > /tmp/pi_tracker.log 2>&1 &")
                        ssh.close(); tm.sleep(2)
                    except: pass

        # USB2
        frame = None
        for idx in [1]:  # skip idx=0
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            tm.sleep(0.05)
            cap.read()
            ret, frame = cap.read(); cap.release()
            if ret and frame.mean() > 10: break

        usb2_r = []
        if frame is not None:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
            gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
            from pupil_apriltags import Detector
            for d in Detector(families="tag36h11", quad_decimate=1.0).detect(gray):
                if d.tag_id in {0,1,2,3}:
                    d.corners*=2.0; d.center=(d.center[0]*2,d.center[1]*2)
                    ok,rv,tv=cv2.solvePnP(self._usb2_obj, d.corners, self._usb2_K, self._usb2_D)
                    if ok:
                        Rt,_=cv2.Rodrigues(rv); tt=tv.reshape(3,1)
                        tw=(self._usb2_Rc@tt+self._usb2_tc).flatten()
                        gsd=np.linalg.norm(self._usb2_R@tw.reshape(3,1)+self._usb2_t)/((self._usb2_K[0,0]+self._usb2_K[1,1])/2)*1000
                        usb2_r.append({"tag_id":int(d.tag_id),"position":tw.tolist(),"gsd":round(float(gsd),2),"source":"USB2"})

        _th.Thread(target=fetch_pi).start()
        while _th.active_count() > 2: tm.sleep(0.005)

        all_r = pi_results + usb2_r
        elapsed = tm.time()-t0
        fps = 1000/elapsed if elapsed > 0 else 0
        self._fps_history.append(fps)
        if len(self._fps_history) > 20: self._fps_history.pop(0)
        avg_fps = sum(self._fps_history)/len(self._fps_history)

        if all_r:
            w = np.array([1.0/max(r["gsd"],0.01) for r in all_r]); w/=w.sum()
            pos = np.zeros(3)
            for wi, r in zip(w, all_r): pos += wi * np.array(r["position"])
            self.cam_status_label.config(text=f"FPS: {avg_fps:.1f} | ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        else:
            self.cam_status_label.config(text=f"FPS: {avg_fps:.1f} | 未检测到")

        wait = max(10, int(80 - elapsed*1000))
        self.root.after(wait, self._live_loop)

    def precision_measure(self):
        """精度验证：直接SSH获取Pi图像+本地USB2，PC统一计算。"""
        self.log("精度测量: 获取三路图像中...")
        def task():
            import json, time as tm, paramiko, os
            t0 = tm.time(); os.makedirs("precision_data", exist_ok=True)
            ts = tm.strftime("%Y%m%d_%H%M%S")
            # SSH Pi 抓图
            try:
                ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
                # PiCamera
                sftp = ssh.open_sftp()
                with sftp.open("/tmp/cap_p.py", "w") as f: f.write('''#!/usr/bin/env python3
import cv2,time
from picamera2 import Picamera2
picam=Picamera2(0)
picam.configure(picam.create_video_configuration(main={'size':(2028,1520),'format':'RGB888'},buffer_count=1))
picam.start();time.sleep(0.3)
cv2.imwrite('/tmp/p.jpg',picam.capture_array());picam.close()''')
                sftp.close()
                ssh.exec_command("python3 /tmp/cap_p.py 2>/dev/null", timeout=15); tm.sleep(1)
                sftp = ssh.open_sftp()
                with sftp.open("/tmp/cap_u.py", "w") as f: f.write('''#!/usr/bin/env python3
import cv2,time
cap=cv2.VideoCapture(0,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2048);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1536)
time.sleep(0.4);[cap.read() for _ in range(4)]
ret,frame=cap.read()
if ret:cv2.imwrite('/tmp/u.jpg',frame);cap.release()''')
                sftp.close()
                ssh.exec_command("python3 /tmp/cap_u.py 2>/dev/null", timeout=15); tm.sleep(1)
                sftp = ssh.open_sftp()
                sftp.get("/tmp/p.jpg", f"precision_data/{ts}_picam.jpg")
                sftp.get("/tmp/u.jpg", f"precision_data/{ts}_usb1.jpg")
                sftp.close(); ssh.close()
            except Exception as e: return f"Pi: {e}"
            # USB2
            frame = None
            try:
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                    tm.sleep(0.3); [cap.read() for _ in range(3)]
                    ret, frame = cap.read(); cap.release()
            except: pass
            if frame is not None and frame.mean()>10:
                cv2.imwrite(f"precision_data/{ts}_usb2.jpg", frame)

            # === PC统一计算三路 ===
            from pupil_apriltags import Detector
            det = Detector(families="tag36h11", quad_decimate=1.0); clahe = cv2.createCLAHE(2.0,(8,8))
            half = 0.0675; obj_pts = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
            cfgs = {
                "PiCam": {"img": f"precision_data/{ts}_picam.jpg",
                    "K": np.array([[1050.3349,0,648.7089],[0,1048.6376,555.0087],[0,0,1]], dtype=np.float64),
                    "dist": np.array([0.132095,-0.532177,0.011064,-0.003189,0.498587], dtype=np.float64)},
                "USB1":  {"img": f"precision_data/{ts}_usb1.jpg",
                    "K": np.array([[1610.2608,0,962.8233],[0,1599.8428,804.8184],[0,0,1]], dtype=np.float64),
                    "dist": np.array([0.150416,-0.251154,0.002832,0.000118,0.133763], dtype=np.float64)},
                "USB2":  {"img": f"precision_data/{ts}_usb2.jpg",
                    "K": np.array([[1997.5587,0,1203.9179],[0,2004.3731,784.2230],[0,0,1]], dtype=np.float64),
                    "dist": np.array([0.08367,-0.15649,0.00321,-0.00835,0.11271], dtype=np.float64)},
            }
            for name in cfgs:
                cfgs[name]["R"] = np.array(self._ext[{"PiCam":"picam_1","USB1":"usb_cam_1","USB2":"usb_cam_2"}[name]]["R"])
                cfgs[name]["t"] = np.array(self._ext[{"PiCam":"picam_1","USB1":"usb_cam_1","USB2":"usb_cam_2"}[name]]["t"]).reshape(3,1)

            all_xy = []; per_cam = {}
            for name, cfg in cfgs.items():
                img = cv2.imread(cfg["img"])
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)
                for d in det.detect(gray):
                    if d.tag_id not in {0,1,2,3}: continue
                    ok,rv,tv = cv2.solvePnP(obj_pts, d.corners, cfg["K"], cfg["dist"])
                    if not ok: continue
                    Rt,_=cv2.Rodrigues(rv); tt=tv.reshape(3,1); Rc=cfg["R"].T; tc=-Rc@cfg["t"]
                    tw = (Rc@tt+tc).flatten()
                    h_loc={0:[1,0,0],1:[0,0,1],2:[-1,0,0],3:[0,0,-1]}.get(d.tag_id,[0,0,1])
                    Rtw=Rc@Rt; hw=Rtw@np.array(h_loc); h2=hw[:2]
                    if np.linalg.norm(h2)>1e-6: h2/=np.linalg.norm(h2)
                    sd=np.array([h2[1],-h2[0]]); sg={0:-1,1:-1,2:1,3:1}.get(d.tag_id,0)
                    off=(h2 if d.tag_id in(1,3)else sd)*sg*0.125
                    center=tw+np.array([off[0],off[1],-0.125])
                    gsd_val=np.linalg.norm(cfg["R"]@tw.reshape(3,1)+cfg["t"])/((cfg["K"][0,0]+cfg["K"][1,1])/2)*1000
                    all_xy.append((center[:2], gsd_val))  # (pos, gsd)
                    per_cam.setdefault(name,[]).append({"tag":int(d.tag_id),"tag_pos":tw.tolist(),"pos":center.tolist(),"gsd":round(float(gsd_val),2)})

            if not all_xy: return "未检测到立方体Tag"
            # GSD加权平均（高GSD相机权重低，避免偏差）
            gsd_weights = np.array([1.0/max(xy[1],0.01) for xy in all_xy])
            gsd_weights /= gsd_weights.sum()
            positions = np.array([xy[0] for xy in all_xy])
            avg_xy = np.average(positions, axis=0, weights=gsd_weights)
            gx = round(avg_xy[0]*2)/2; gy = round(avg_xy[1]*2)/2
            gx = max(0, min(4.5, gx)); gy = max(0, min(5.0, gy))
            self.root.after(0, lambda: self.gt_label.config(text=f"({gx:.1f}, {gy:.1f})m"))
            # 误差
            for name in per_cam:
                for m in per_cam[name]:
                    m["err_cm"] = round(float(np.linalg.norm(np.array(m["pos"][:2])-[gx,gy])*100),2)
            record = {"time": ts, "ground_truth": [gx, gy], "measured_avg": [round(float(avg_xy[0]),3), round(float(avg_xy[1]),3)],
                      "camera_results": {k: [{kk:vv[kk] for kk in vv} for vv in per_cam[k]] for k in per_cam}}
            with open(f"precision_data/{ts}.json","w") as f: json.dump(record, f, indent=2)
            lines = [f"精度测量: 真值=({gx:.1f},{gy:.1f})m"]
            for src in sorted(per_cam.keys()):
                for m in per_cam[src][:2]:
                    tp=m["tag_pos"]; cp=m["pos"]
                    lines.append(f"  {src} Tag{m['tag']}: Tag({tp[0]:.3f},{tp[1]:.3f}) -> 中心({cp[0]:.3f},{cp[1]:.3f}) err={m['err_cm']:.1f}cm")
            lines.append(f"  保存+3原图 | {(tm.time()-t0)*1000:.0f}ms")
            files = [f for f in os.listdir("precision_data") if f.endswith(".json")]
            self.root.after(0, lambda: self.precision_count.config(text=f"已测: {len(files)} 次"))
            if hasattr(self,"bev_image") and self.bev_image:
                self.root.after(0, lambda: self._draw_precision_point(gx, gy))
            return "\n".join(lines)
        self._run_in_thread(task, lambda m: self.log(m))

    def clear_precision(self):
        """删除所有精度测量数据。"""
        import shutil, os
        if os.path.exists("precision_data"):
            n = len([f for f in os.listdir("precision_data") if f.endswith(".json")])
            shutil.rmtree("precision_data")
            os.makedirs("precision_data", exist_ok=True)
            self.precision_count.config(text="已测: 0 次")
            self.gt_label.config(text="---")
            if hasattr(self, "bev_canvas"): self.bev_canvas.delete("prec")
            self.log(f"已清除 {n} 次测量数据")

    def precision_report(self):
        """从 precision_data/ 生成综合精度报告。"""
        import json as _j, os
        files = sorted([f for f in os.listdir("precision_data") if f.endswith(".json")])
        if not files: self.log("无精度数据"); return

        all_data = []
        for fn in files:
            with open(f"precision_data/{fn}") as f: all_data.append(_j.load(f))

        # 统计
        lines = [f"=== 精度报告 ({len(all_data)} 次测量) ==="]
        lines.append(f"{'真值':12s} | {'相机':8s} | {'Tag':3s} | {'测得位置':20s} | {'误差cm':7s} | {'GSD':5s}")
        lines.append("-"*75)

        cam_errs = {}
        for d in all_data:
            gt = d["ground_truth"]
            for cam, measures in d["camera_results"].items():
                cam_errs.setdefault(cam, [])
                for m in measures:
                    err = m["err_cm"]
                    cam_errs[cam].append(err)
                    lines.append(f"({gt[0]:.2f},{gt[1]:.2f})    | {cam:8s} | T{m['tag']:2d} | ({m['pos'][0]:.3f},{m['pos'][1]:.3f},{m['pos'][2]:.3f}) | {err:5.1f}cm | {m['gsd']:4.1f}")

        lines.append("-"*75)
        lines.append("汇总:")
        for cam, errs in sorted(cam_errs.items()):
            lines.append(f"  {cam}: 平均={np.mean(errs):.2f}cm  最大={np.max(errs):.2f}cm  n={len(errs)}")

        # 保存HTML报告
        rows = "".join(f"<tr><td>{d['time']}</td><td>({d['ground_truth'][0]:.2f},{d['ground_truth'][1]:.2f})</td>"
                       f"<td>{len(d['camera_results'])}</td></tr>" for d in all_data)
        html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>精度报告</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#e94560}}table{{border-collapse:collapse;width:100%}}th{{background:#0f3460;padding:8px}}td{{padding:6px;border-bottom:1px solid #333}}</style></head><body>
<h1>精度验证报告 ({len(all_data)}次)</h1><pre style="background:#111;padding:12px;font-size:12px">{"".join(l+'<br>' for l in lines)}</pre>
<table><tr><th>时间</th><th>真值</th><th>相机数</th></tr>{rows}</table></body></html>'''
        with open("precision_report.html","w",encoding="utf-8") as f: f.write(html)
        self.log(f"报告: precision_report.html | " + " | ".join(f"{c}:avg={np.mean(e):.1f}cm" for c,e in sorted(cam_errs.items())))
        import webbrowser; webbrowser.open("precision_report.html")

    def _draw_precision_point(self, gx, gy):
        """在BEV上标出精度测量点（Canvas叠加）。"""
        # BEV参数需与cart_report.py一致
        X_MIN, X_MAX = 0.0, 4.5; Y_MIN, Y_MAX = -0.5, 5.0; PPM = 200; BM = 40
        BH = int((Y_MAX-Y_MIN)*PPM)+2*BM; BW = int((X_MAX-X_MIN)*PPM)+2*BM
        def b2p(x, y): return (BM+int((x-X_MIN)*PPM), BH-BM-int((y-Y_MIN)*PPM))
        # Canvas坐标需要缩放到当前显示大小
        cw = self.bev_canvas.winfo_width() or BW
        ch = self.bev_canvas.winfo_height() or BH
        sx, sy = cw/BW, ch/BH
        u, v = b2p(gx, gy)
        u, v = int(u*sx), int(v*sy)
        # 画十字标记
        r = 8
        self.bev_canvas.create_line(u-r, v, u+r, v, fill="#ffff00", width=2, tags="prec")
        self.bev_canvas.create_line(u, v-r, u, v+r, fill="#ffff00", width=2, tags="prec")
        self.bev_canvas.create_text(u+14, v-10, text=f"({gx:.1f},{gy:.1f})", fill="#ffff00", font=("",8), anchor=tk.W, tags="prec")

    def export_report(self):
        dirs = sorted(Path(".").glob("tracking_run_*"), reverse=True)
        if dirs and (dirs[0]/"cart_tracking_report.html").exists():
            import webbrowser; webbrowser.open(str(dirs[0]/"cart_tracking_report.html"))
            self.log("已打开报告")
        else: messagebox.showwarning("导出","没有找到报告，请先执行一次定位")

def main():
    root = tk.Tk(); App(root); root.mainloop()

if __name__ == "__main__": main()
