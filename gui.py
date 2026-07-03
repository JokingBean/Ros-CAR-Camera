#!/usr/bin/env python3
"""GUI 控制台 — 从 src/ 加载"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.dirname(__file__))
from gui_app import main as gui_main
gui_main()
