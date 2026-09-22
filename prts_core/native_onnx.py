"""Actual C ABI ONNX adapter for parity checks; shares the caller's ORT library."""
import ctypes as C
from pathlib import Path
import numpy as np


class Tensor(C.Structure):
    _fields_=[('data',C.POINTER(C.c_float)),('element_count',C.c_int64),('rank',C.c_int32),('shape',C.c_int64*8)]


class NativeONNX:
    def __init__(self,bridge,runtime,model,threads=4,use_coreml=False):
        self.runtime=C.CDLL(str(Path(runtime).resolve()))
        self.runtime.OrtGetApiBase.argtypes=[];self.runtime.OrtGetApiBase.restype=C.c_void_p
        self.lib=C.CDLL(str(Path(bridge).resolve()))
        self.lib.prts_onnx_create.argtypes=[C.c_void_p,C.c_char_p,C.c_int32,C.c_int32]
        self.lib.prts_onnx_create.restype=C.c_void_p
        self.lib.prts_onnx_run.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_int64,C.POINTER(C.c_int64),C.c_int32,C.POINTER(Tensor)]
        self.lib.prts_onnx_run.restype=C.c_int32
        self.lib.prts_onnx_last_error.argtypes=[];self.lib.prts_onnx_last_error.restype=C.c_char_p
        self.lib.prts_onnx_destroy.argtypes=[C.c_void_p];self.lib.prts_onnx_destroy.restype=None
        self.handle=self.lib.prts_onnx_create(self.runtime.OrtGetApiBase(),str(Path(model).resolve()).encode('utf-8'),threads,int(use_coreml))
        if not self.handle:raise RuntimeError(self.error())

    def error(self):return self.lib.prts_onnx_last_error().decode('utf-8',errors='replace')

    def run(self,value):
        x=np.ascontiguousarray(value,np.float32);shape=(C.c_int64*x.ndim)(*x.shape);out=Tensor()
        code=self.lib.prts_onnx_run(self.handle,x.ctypes.data_as(C.POINTER(C.c_float)),x.size,shape,x.ndim,C.byref(out))
        if code:raise RuntimeError(self.error())
        return np.ctypeslib.as_array(out.data,(out.element_count,)).copy().reshape(list(out.shape)[:out.rank])

    def close(self):
        if self.handle:self.lib.prts_onnx_destroy(self.handle);self.handle=None
