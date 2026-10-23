"""Print this process's memory (MB) after each heavy import. Windows only."""
import ctypes
import ctypes.wintypes as w


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD), ("PageFaultCount", w.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
kernel32.GetCurrentProcess.restype = w.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]


def rss_mb() -> int:
    info = PMC()
    info.cb = ctypes.sizeof(info)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb)
    return round(info.WorkingSetSize / 1048576)


print("baseline", rss_mb(), "MB")
for name in ("numpy", "fastapi", "pydantic_settings", "httpx", "pypdf", "mcp", "langgraph.graph", "fastembed"):
    __import__(name)
    print(f"after {name}: {rss_mb()} MB")
