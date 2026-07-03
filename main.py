#!/usr/bin/env python3
"""
ROS-Camera 三相机 BEV 系统 — 主菜单
====================================
python main.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def menu():
    while True:
        print()
        print("=" * 40)
        print("  ROS-Camera 三相机 BEV")
        print("=" * 40)
        print("  1. BEV 俯视图（标定 → 融合 → 报告）")
        print("  2. 精度测试（逐点定位 + 误差）")
        print("  3. 连续定位（实时 FPS + XY）")
        print("  4. 退出")
        print()

        choice = input("  > ").strip()
        if choice == "1":
            from drivers.bev import main
            main()
        elif choice == "2":
            from drivers.precision import main
            main()
        elif choice == "3":
            from drivers.live import main
            main()
        elif choice == "4":
            print("  bye")
            break
        else:
            print("  无效")

if __name__ == "__main__":
    menu()
