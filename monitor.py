# !/usr/bin/python
# coding=utf-8

import os
import psutil


def get_current_memory_gb() -> int:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.
