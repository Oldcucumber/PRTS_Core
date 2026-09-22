"""ctypes reference binding for the portable geometry C ABI; no model runtime."""
import ctypes as C
from pathlib import Path
import numpy as np


class NativeGeometry:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve()))
        byte=C.POINTER(C.c_uint8);integer=C.POINTER(C.c_int32);real=C.POINTER(C.c_double)
        self.lib.prts_corridor_search.argtypes=[byte,C.c_int32,C.c_int32,C.c_int32,C.c_int32,integer,C.c_int32,integer,C.c_int32,real]
        self.lib.prts_corridor_search.restype=C.c_int32
        self.lib.prts_path_validate.argtypes=[byte,C.c_int32,C.c_int32,integer,C.c_int32,integer]
        self.lib.prts_path_validate.restype=C.c_int32
        self.lib.prts_path_heading.argtypes=[integer,C.c_int32,C.c_int32,C.c_int32,real]
        self.lib.prts_path_heading.restype=C.c_int32

    @staticmethod
    def ptr(array,ctype):return array.ctypes.data_as(C.POINTER(ctype))

    def search(self,free,anchor=None,prefer=None,prior=None):
        mask=np.ascontiguousarray(free,np.uint8);h,w=mask.shape
        previous=np.ascontiguousarray(prior if prior else [],np.int32).reshape(-1,2)
        out=np.empty((h*w,2),np.int32);coverage=C.c_double()
        n=self.lib.prts_corridor_search(self.ptr(mask,C.c_uint8),w,h,-1 if anchor is None else anchor,
            prefer or 0,self.ptr(previous,C.c_int32),len(previous),self.ptr(out,C.c_int32),h*w,C.byref(coverage))
        if n<0:raise RuntimeError(f'Native path error {n}')
        return [tuple(map(int,p)) for p in out[:n]],coverage.value

    def validate(self,points,free):
        mask=np.ascontiguousarray(free,np.uint8);h,w=mask.shape
        xy=np.ascontiguousarray(points,np.int32).reshape(-1,2);checked=C.c_int32()
        outside=self.lib.prts_path_validate(self.ptr(mask,C.c_uint8),w,h,self.ptr(xy,C.c_int32),len(xy),C.byref(checked))
        if outside<0:raise RuntimeError(f'Native validation error {outside}')
        return dict(valid=checked.value>0 and outside==0,checked_pixels=checked.value,outside_free_pixels=outside)

    def heading(self,points,shape):
        h,w=shape;xy=np.ascontiguousarray(points,np.int32).reshape(-1,2);angles=(C.c_double*3)()
        code=self.lib.prts_path_heading(self.ptr(xy,C.c_int32),len(xy),w,h,angles)
        if code<0:raise RuntimeError(f'Native heading error {code}')
        return dict(zip(('heading_deg','near_heading_deg','far_heading_deg'),angles))
