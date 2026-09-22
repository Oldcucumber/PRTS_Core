"""Image-sector obstacle monitoring when no connected sidewalk is available."""
import math
import cv2
import numpy as np


def forward_scan(perception, half_angle_deg=15., horizontal_fov_deg=60.):
    # Until calibrated intrinsics arrive, angles explicitly use an assumed FOV.
    fraction=math.tan(math.radians(half_angle_deg))/math.tan(math.radians(horizontal_fov_deg/2))
    left,right=max(0.,.5-fraction/2),min(1.,.5+fraction/2)
    top=.55
    obstacles=[]
    for item in perception.get('detections',[]):
        if not item.get('blocking',True) or item.get('score',1.)<.35:continue
        x1,y1,x2,y2=item['box']
        overlap=max(0.,min(right,x2)-max(left,x1))*max(0.,min(1.,y2)-max(top,y1))
        if overlap/((right-left)*(1-top))>=.01:
            obstacles.append(dict(label=item['label'],box=item['box'],source='detector'))
    groups=perception.get('semantic_groups')
    if groups is not None:
        h,w=groups.shape
        sector=np.zeros((h,w),bool);sector[int(top*h):,int(left*w):int(math.ceil(right*w))]=True
        confidence=perception.get('confidence',np.ones((h,w),np.float32))
        occupied=sector & np.isin(groups,[3,4]) & (confidence>=.60)
        count,components,stats,_=cv2.connectedComponentsWithStats(occupied.astype(np.uint8),8)
        minimum=max(8,int(sector.sum()*.01))
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<minimum:continue
            x,y,bw,bh=map(int,stats[i,:4])
            labels=['固定或动态障碍']
            if 'classes' in perception:
                ids=np.unique(perception['classes'][components==i])
                labels=[perception['labels'].get(int(cid),str(cid)) for cid in ids]
            obstacles.append(dict(label='/'.join(labels),box=[x/w,y/h,(x+bw)/w,(y+bh)/h],source='semantic'))
    return dict(half_angle_deg=half_angle_deg,horizontal_fov_deg=horizontal_fov_deg,
        angle_source='assumed_horizontal_fov',coordinate_space='image_normalized',
        sector_box=[left,top,right,1.],obstacles=obstacles,
        assessment='obstacle_detected' if obstacles else 'no_obstacle_detected',
        metric_distance_available=False)
