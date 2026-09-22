"""SANPO human-label metrics, isolated from model/runtime imports."""
import cv2
import numpy as np


def label_metrics(p,path):
    raw=cv2.imread(str(path),cv2.IMREAD_COLOR)
    gt=cv2.resize(raw[:,:,2],(p['classes'].shape[1],p['classes'].shape[0]),interpolation=cv2.INTER_NEAREST)
    valid=gt!=0;truth=np.isin(gt,[3,6,17]);pred=p['sidewalk_class']&(p['sidewalk']>=.5)
    tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'walkable_iou':tp/max(1,tp+fp+fn),'walkable_truth_pixels':int(truth.sum()),
            'roadway_admitted_pixels':int((pred&np.isin(gt,[1,5,19])).sum()),
            'fixed_admitted_pixels':int((pred&np.isin(gt,[2,4,7,8,9,10,11,15,16,18,20,22,23,24,25,26,28,29])).sum()),
            'dynamic_admitted_pixels':int((pred&np.isin(gt,[12,13,14,21])).sum())}
