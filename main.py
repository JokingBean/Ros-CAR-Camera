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
        print("  ROS-Camera 三相机 BEV 系统")
        print("=" * 40)
        print("  1. BEV 俯视图（抓图 → 标定 → 融合 → 报告）")
        print("  2. 精度测试（立方体定位误差测量）")
        print("  3. 退出")
        print()

        choice = input("  选择 [1-3]: ").strip()
        if choice == "1":
            from drivers.bev import main
            main()
        elif choice == "2":
            from drivers.precision import main
            main()
        elif choice == "3":
            print("  退出")
            break
        else:
            print("  无效选择，请重试")


if __name__ == "__main__":
    menu()
