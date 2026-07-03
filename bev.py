#!/usr/bin/env python3
"""一键 BEV — 从 src/ 加载"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.dirname(__file__))
from run_all import main
main()
