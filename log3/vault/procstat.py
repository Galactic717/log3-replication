"""Ресурси поточного процесу: процесорний час і піковий резидентний обсяг пам'яті.

Працює на Linux, macOS і Windows (модуля resource у Windows нема). Процес міряє сам себе,
тому CLI із LOG3_VAULT_RUSAGE=1 додає ці числа у свій JSON, а вимірювання не залежать від
того, як ОС рахує ресурси дочірніх процесів.
"""
from __future__ import annotations

import os
import sys
import time


def peak_rss_kib() -> int | None:
    """Піковий RSS процесу в КіБ або None, якщо ОС його не дає."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *((n, ctypes.c_size_t) for n in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                            "PeakPagefileUsage"))]

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD)
        c = _Counters()
        c.cb = ctypes.sizeof(c)
        return c.PeakWorkingSetSize // 1024 if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), c, c.cb) else None
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if sys.platform == "darwin" else rss     # macOS дає байти, Linux — КіБ


def self_usage() -> dict:
    return {"cpu_seconds": round(time.process_time(), 4), "peak_rss_kib": peak_rss_kib()}
