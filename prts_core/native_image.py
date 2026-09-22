"""Portable image operations, separately audited against the OpenCV reference."""
import ctypes as C
from pathlib import Path
import numpy as np


class NativeImage:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve()));pf=C.POINTER(C.c_float);pi=C.POINTER(C.c_int32);pb=C.POINTER(C.c_uint8)
        self.lib.prts_letterbox_rgb.argtypes=[pb,C.c_int32,C.c_int32,C.c_int32,C.c_int32,pf,pf,pf,pi]
        self.lib.prts_letterbox_rgb.restype=C.c_int32
        self.lib.prts_decode_semantic.argtypes=[pf,C.c_int32,C.c_int32,C.c_int32,pi,C.c_int32,C.c_int32,C.c_int32,C.c_int32,pi,C.c_int32,pb,pf,pf,pb]
        self.lib.prts_decode_semantic.restype=C.c_int32

    @staticmethod
    def p(array,kind):return array.ctypes.data_as(C.POINTER(kind))

    def letterbox(self,rgb,width,height,mean=None,std=None):
        rgb=np.ascontiguousarray(rgb,np.uint8);h,w=rgb.shape[:2]
        mean=np.asarray(mean if mean is not None else [0,0,0],np.float32)
        std=np.asarray(std if std is not None else [1,1,1],np.float32)
        out=np.empty((1,3,height,width),np.float32);crop=np.empty(4,np.int32)
        code=self.lib.prts_letterbox_rgb(self.p(rgb,C.c_uint8),w,h,width,height,self.p(mean,C.c_float),
            self.p(std,C.c_float),self.p(out,C.c_float),self.p(crop,C.c_int32))
        if code:raise ValueError('Native letterbox input invalid')
        return out,tuple(map(int,crop))

    def decode(self,scores,crop,input_width,input_height,grid_width,grid_height,walk_ids):
        scores=np.ascontiguousarray(scores,np.float32);nc,sh,sw=scores.shape
        crop=np.ascontiguousarray(crop,np.int32);ids=np.ascontiguousarray(walk_ids,np.int32)
        classes=np.empty((grid_height,grid_width),np.uint8);walk=np.empty_like(classes,np.float32)
        confidence=np.empty_like(walk);walkable=np.empty_like(classes)
        code=self.lib.prts_decode_semantic(self.p(scores,C.c_float),nc,sw,sh,self.p(crop,C.c_int32),input_width,input_height,
            grid_width,grid_height,self.p(ids,C.c_int32),len(ids),self.p(classes,C.c_uint8),self.p(walk,C.c_float),
            self.p(confidence,C.c_float),self.p(walkable,C.c_uint8))
        if code:raise ValueError('Native semantic dimensions invalid')
        return dict(classes=classes,sidewalk=walk,confidence=confidence,walkable=walkable.astype(bool))
