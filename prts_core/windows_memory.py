"""Windows-only measurement helper; not required by model inference or Apple code."""
import ctypes as C
from pathlib import Path


class WindowsGPUMemory:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.prts_gpu_memory.argtypes=[C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
        self.lib.prts_gpu_memory.restype=C.c_int32
    def sample(self):
        local=C.c_uint64();nonlocal_=C.c_uint64()
        status=self.lib.prts_gpu_memory(C.byref(local),C.byref(nonlocal_))
        if status<0:raise RuntimeError(f'DXGI memory query failed: {status & 0xffffffff:08x}')
        return dict(gpu_local_usage=local.value,gpu_nonlocal_usage=nonlocal_.value)
