"""Semantic groups retain the model's raw labels; unknown classes stay unknown."""
import numpy as np

GROUP_NAMES={0:'unknown',1:'walkable',2:'roadway',3:'fixed_obstacle',4:'dynamic_object'}
WALKABLE={'sidewalk','flat-sidewalk','pedestrian area','paved trail','other walkable surface'}
ROAD={'road','flat-road','service lane','bike lane','parking','crosswalk - plain','lane marking - crosswalk','lane marking - general','rail track','railway track'}
DYNAMIC={'person','pedestrian','rider','bicyclist','motorcyclist','other rider','bird','animal','ground animal','car','bus','truck','train','vehicle','motorcycle','bicycle','boat','caravan','on rails','other vehicle','trailer','wheeled slow','ego vehicle'}
FIXED={'curb','fence','wall/fence','wall','guard rail','guard rail/road barrier','barrier','building','bridge','tunnel','vegetation','tree','tree trunk','water','water body','stairs','bench','bike rack','billboard','banner','catch basin','cctv camera','fire hydrant','junction box','mailbox','phone booth','pothole','street light','pole','utility pole','traffic light','traffic sign','traffic sign frame','traffic sign (back)','traffic sign (front)','trash can','obstacle','hand rail','opening-door','opening-gate','bus stop'}
ROAD.update({'flat-crosswalk','flat-cyclinglane','flat-parkingdriveway','flat-railtrack'})
FIXED.update({'flat-curb','construction-building','construction-door','construction-wall','construction-fenceguardrail',
              'construction-bridge','construction-tunnel','construction-stairs','object-pole','object-trafficsign',
              'object-trafficlight','nature-vegetation'})
DYNAMIC.update({'human-person','human-rider','vehicle-car','vehicle-truck','vehicle-bus','vehicle-tramtrain',
                'vehicle-motorcycle','vehicle-bicycle','vehicle-caravan','vehicle-cartrailer'})


def group_labels(classes,labels):
    lookup=np.zeros(256,np.uint8)
    for idx,label in labels.items():
        key=label.casefold()
        lookup[int(idx)]=1 if key in WALKABLE else 2 if key in ROAD else 4 if key in DYNAMIC else 3 if key in FIXED else 0
    return lookup[classes]


def region_records(classes,labels,min_pixels=20):
    """Bounded display contours with holes; the dense mask remains authoritative.

    Regions classify visible surfaces, not occupied 3D ground. A vegetation
    region is never relabelled as a flowerbed, and sky/terrain remain unknown.
    """
    import cv2
    h,w=classes.shape;regions=[]
    groups=group_labels(classes,labels)
    for cid in np.unique(classes):
        mask=(classes==cid).astype(np.uint8)
        count,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for idx in range(1,count):
            area=int(stats[idx,cv2.CC_STAT_AREA])
            if area<min_pixels:continue
            component=(components==idx).astype(np.uint8)
            contours,hierarchy=cv2.findContours(component,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            polygons=[];holes=[]
            for contour,relation in zip(contours,hierarchy[0]):
                pts=cv2.approxPolyDP(contour,.65,True).reshape(-1,2)
                poly=[[float(x/w),float(y/h)] for x,y in pts]
                (holes if relation[3]>=0 else polygons).append(poly)
            x,y,bw,bh=map(int,stats[idx,:4]);gid=int(groups[components==idx][0])
            regions.append({'class_id':int(cid),'label':labels[int(cid)],'group':GROUP_NAMES[gid],
                'area_fraction':area/(h*w),'box':[x/w,y/h,(x+bw)/w,(y+bh)/h],
                'contours':polygons,'holes':holes,'coordinate_space':'image_normalized',
                'geometry':'display_contours_use_dense_mask_for_validation'})
    return sorted(regions,key=lambda r:r['area_fraction'],reverse=True)[:64]


def semantic_view(bgr,perception,legend=True):
    import cv2
    h,w=bgr.shape[:2]
    roles=cv2.resize(perception['semantic_groups'],(w,h),interpolation=cv2.INTER_NEAREST)
    colors=np.array([[90,90,90],[45,185,70],[190,130,40],[35,100,215],[200,70,180]],np.uint8)
    view=(bgr*.5+colors[roles]*.5).astype(np.uint8)
    if legend:
        scale=max(.6,w/1600)
        cv2.rectangle(view,(0,h-int(33*scale)),(w,h),(12,16,20),-1)
        for i,name in GROUP_NAMES.items():
            x=int((12+i*w/5));y=h-int(10*scale)
            cv2.putText(view,name,(x,y),0,.5*scale,tuple(map(int,colors[i])),max(1,round(scale)))
    return view
