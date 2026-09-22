"""OCR-only runtime, independent of the PyTorch research model module."""
from pathlib import Path
import numpy as np


class LocalOCR:
    def __init__(self,threads=4,profile='bounded_upright',model_dir=None):
        import rapidocr
        root=Path(model_dir) if model_dir is not None else Path(rapidocr.__file__).parent/'models'
        files={'Det.model_path':root/'PP-OCRv6_det_small.onnx',
               'Rec.model_path':root/'PP-OCRv6_rec_small.onnx',
               'Cls.model_path':root/'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
        for p in files.values():
            if not p.exists():raise FileNotFoundError(p)
        settings={'Global.max_side_len':960,'Det.limit_type':'max','Global.use_cls':False} if profile.startswith('bounded_upright') else {}
        if profile=='bounded_upright_no_arena':
            settings['EngineConfig.onnxruntime.enable_cpu_mem_arena']=False
        self.engine=rapidocr.RapidOCR(params={**{k:str(v) for k,v in files.items()},**settings,
            'Global.log_level':'warning','EngineConfig.onnxruntime.intra_op_num_threads':threads,
            'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':960})

    def __call__(self,bgr):
        r=self.engine(bgr)
        if r.txts is None:return []
        return [dict(text=str(t),score=float(s),box=np.asarray(b).tolist())
                for t,s,b in zip(r.txts,r.scores,r.boxes)]
