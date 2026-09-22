"""Connected image-space corridors and turn hysteresis; no metric clearance claim."""
import heapq
import math
import cv2
import numpy as np


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def inset_corridor(free):
    h,w=free.shape;inset=np.zeros_like(free,dtype=bool)
    for y in range(h):
        radius=max(1,round(w*(.004+.046*max(0,(y/h-.45)/.50))))
        inset[y]=cv2.erode(free[y:y+1].astype(np.uint8),np.ones((1,2*radius+1),np.uint8),
                           borderType=cv2.BORDER_CONSTANT,borderValue=0)[0].astype(bool)
    return inset


def validate_path(points,free):
    """Rasterize every segment and endpoint, not only sampled vertices."""
    h,w=free.shape;pixels=set()
    for a,b in zip(points,points[1:]):
        n=max(abs(b[0]-a[0]),abs(b[1]-a[1]),1)
        for t in range(n+1):pixels.add((round(a[0]+(b[0]-a[0])*t/n),round(a[1]+(b[1]-a[1])*t/n)))
    if len(points)==1:pixels.add(tuple(points[0]))
    outside=sum(not (0<=x<w and 0<=y<h and free[y,x]) for x,y in pixels)
    return {'valid':bool(pixels) and outside==0,'checked_pixels':len(pixels),'outside_free_pixels':outside,
            'scope':'current predicted image mask; not physical safety or ground-truth correctness'}


def search_corridor(free,anchor=None,prefer=None,prior=None):
    """A* in the near-field connected component, with no diagonal corner cutting.

    prefer is a lateral preference (-1/0/+1), never a map-to-image projection.
    Registration may softly favour a previous path, but cannot add free pixels.
    """
    h,w=free.shape;inset=inset_corridor(free)
    sy=int(h*.94);ey=int(h*.42);xs=np.arange(w)
    if anchor is None:
        candidates=xs[inset[sy]&(np.abs(xs-w/2)<w*.08)]
        if not len(candidates):return [],0.
        anchor=int(candidates[np.argmin(np.abs(candidates-w/2))])
    if not 0<=anchor<w or not inset[sy,anchor]:return [],0.
    inset[:ey]=False;inset[sy+1:]=False
    _,components=cv2.connectedComponents(inset.astype(np.uint8),connectivity=4)
    connected=components==components[sy,anchor]
    gy=int(np.nonzero(connected)[0].min());coverage=(sy-gy)/max(1,sy-ey)
    if gy==sy:return [],0.
    # Zero border also gives finite clearance for an entirely free image.
    clearance=cv2.distanceTransform(np.pad(free.astype(np.uint8),1),cv2.DIST_L2,3)[1:-1,1:-1]
    candidates=np.flatnonzero(connected[gy]);target=anchor if not prefer else w*(.5+.26*prefer)
    score=clearance[gy,candidates]/w-.20*np.abs(candidates-target)/w
    gx=int(candidates[np.argmax(score)]);prior_distance=None
    if prior:
        line=np.ones((h,w),np.uint8);cv2.polylines(line,[np.asarray(prior,np.int32)],False,0,1)
        prior_distance=cv2.distanceTransform(line,cv2.DIST_L2,3)/w
    cost=np.full((h,w),np.inf,np.float64);parents=np.full((h,w,2),-1,np.int16)
    cost[sy,anchor]=0.;queue=[(math.hypot(gx-anchor,sy-gy),0.,anchor,sy)]
    neighbours=((-1,-1,1.4142135623730951),(0,-1,1.),(1,-1,1.4142135623730951),(-1,0,1.),(1,0,1.),(-1,1,1.4142135623730951),(0,1,1.),(1,1,1.4142135623730951))
    reached=False
    while queue:
        _,g,x,y=heapq.heappop(queue)
        if g>cost[y,x]+1e-8:continue
        if x==gx and y==gy:reached=True;break
        for dx,dy,d in neighbours:
            nx,ny=x+dx,y+dy
            if not (0<=nx<w and ey<=ny<=sy and connected[ny,nx]):continue
            if dx and dy and not (inset[y,nx] and inset[ny,x]):continue
            extra=.8/(1+float(clearance[ny,nx]))
            if prior_distance is not None:extra+=.15*min(.3,float(prior_distance[ny,nx]))
            ng=g+d*(1+extra)
            if ng+1e-8<cost[ny,nx]:
                cost[ny,nx]=ng;parents[ny,nx]=(x,y)
                heapq.heappush(queue,(ng+math.hypot(gx-nx,gy-ny),ng,nx,ny))
    if not reached:return [],0.
    points=[(gx,gy)];x,y=gx,gy
    while (x,y)!=(anchor,sy):
        x,y=map(int,parents[y,x]);points.append((x,y))
    points.reverse()
    return points,float(coverage)


def path_heading(points,shape):
    """Fixed lookahead rows avoid direction depending on truncated path length."""
    h,w=shape
    def at(row):
        distance=min(abs(y-row*h) for x,y in points)
        return np.median([x for x,y in points if abs(y-row*h)<=distance+1])/w
    near=.87;far=max(.57,min(y for x,y in points)/h);middle=(near+far)/2
    heading=math.degrees(math.atan2(at(far)-at(near),near-far))
    near_heading=math.degrees(math.atan2(at(middle)-at(near),near-middle))
    far_heading=math.degrees(math.atan2(at(far)-at(middle),middle-far))
    return {'heading_deg':heading,'near_heading_deg':near_heading,'far_heading_deg':far_heading,
            'lookahead_rows':[near,far],'coordinate_space':'image_normalized',
            'meaning':'path relative to image vertical; not compass heading or body velocity'}


def estimate_motion(previous,gray,dt):
    result={'valid':False,'quality':0.,'tracked_points':0,'inlier_fraction':0.,
            'translation_fraction':0.,'rotation_deg':0.,'scale':1.,'reset':False,
            'kind':'visual_affine_image_motion','body_motion_available':False}
    if previous is None:return result,None
    if dt<=0 or dt>2:result['reset']=True;return result,None
    points=cv2.goodFeaturesToTrack(previous,160,.015,7)
    if points is None or len(points)<12:return result,None
    moved,ok,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(21,21),maxLevel=3)
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,previous,moved,None,winSize=(21,21),maxLevel=3)
    valid=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
    a,b=points[valid],moved[valid];result['tracked_points']=len(a)
    if len(a)<12:return result,None
    affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if affine is None:return result,None
    fraction=float(inliers.mean());quality=fraction*min(1.,len(a)/60)
    scale=float(np.hypot(affine[0,0],affine[0,1]));rotation=float(np.degrees(np.arctan2(affine[1,0],affine[0,0])))
    displacement=float(np.median(np.linalg.norm(b-a,axis=2)))/gray.shape[1]
    result.update(valid=quality>=.45,quality=quality,inlier_fraction=fraction,translation_fraction=displacement,
                  rotation_deg=rotation,scale=scale,reset=displacement>.10 or abs(rotation)>10 or not .83<scale<1.2)
    return result,affine if result['valid'] and not result['reset'] else None


class Guide:
    def __init__(self,temporal=True,forward_half_angle_deg=15.,horizontal_fov_deg=60.):
        self.forward_half_angle_deg=forward_half_angle_deg;self.horizontal_fov_deg=horizontal_fov_deg
        self.temporal=temporal;self.previous=[];self.last_time=None;self.last_gray=None
        self.last_direction=None;self.pending=None;self.pending_since=None;self.pending_count=0
        self.stable=0;self.last_path=[];self.last_shape=None

    def reset(self):self.__init__(self.temporal,self.forward_half_angle_deg,self.horizontal_fov_deg)

    def __call__(self,bgr,perception,time_s,route_hint=None):
        ground=perception['sidewalk_class']&(perception['sidewalk']>=.50)
        h,w=ground.shape;gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(240,135))
        dt=0 if self.last_time is None else time_s-self.last_time
        estimate,affine=estimate_motion(self.last_gray,gray,dt)
        reset=estimate['reset'] or (self.last_time is not None and dt<=0)
        # A long inference gap invalidates history, not a fresh valid image.
        # Moderate camera turns reset registration so the new geometry can win.
        reject_current=(self.last_time is not None and dt<=0) or (estimate['valid'] and (
            estimate['translation_fraction']>.20 or abs(estimate['rotation_deg'])>15 or not .70<estimate['scale']<1.4))
        if reset:
            self.previous=[];self.last_direction=None;self.pending=None;self.stable=0;self.last_path=[]
        prior=None
        if self.temporal and affine is not None and self.last_path and self.last_shape==(h,w):
            old=np.asarray(self.last_path,np.float32)*np.array([240/w,135/h],np.float32)
            mapped=cv2.transform(old[None],affine)[0]*np.array([w/240,h/135])
            prior=[tuple(map(int,np.rint(p))) for p in mapped]
        self.last_gray=gray;self.last_time=time_s
        detections=[dict(d) for d in perception.get('detections',[])];matched=set()
        for d in detections:
            d['approaching']=False
            matches=[(iou(d['box'],p['box']),j,p) for j,p in enumerate(self.previous)
                     if d['class_id']==p['class_id'] and j not in matched]
            if matches:
                score,j,old=max(matches,key=lambda x:x[0])
                if score>.25 and 0<dt<2:
                    matched.add(j);a,b=d['box'],old['box']
                    growth=(np.sqrt((a[2]-a[0])*(a[3]-a[1]))-np.sqrt((b[2]-b[0])*(b[3]-b[1])))/dt
                    d['approaching']=bool(growth>.09 and a[3]>.55)
            d['seen_s']=time_s
        self.previous=detections;blocked=np.zeros((h,w),np.uint8)
        for d in detections:
            if not d.get('blocking',True):continue
            x1,y1,x2,y2=d['box'];pad=.006+.045*(y2-y1)
            a=max(0,int((x1-pad)*w));b=min(w,int(np.ceil((x2+pad)*w)))
            c=max(0,int((y1-.01)*h));e=min(h,int(np.ceil((y2+.025)*h)))
            blocked[c:e,a:b]=1
        free=ground&~blocked.astype(bool);prefer=None;route_state='absent'
        if route_hint:
            bearing=route_hint.get('relative_bearing_deg')
            if route_hint.get('status')!='following':route_state='inactive'
            elif bearing is None or not np.isfinite(bearing):route_state='camera_heading_unavailable'
            elif abs(bearing)>75:route_state='outside_forward_view'
            else:
                prefer=-1 if bearing<-15 else 1 if bearing>15 else 0;route_state='lateral_preference_only'
        baseline,base_coverage=search_corridor(ground,prefer=prefer)
        anchor=baseline[0][0] if baseline else None
        points,coverage=search_corridor(free,anchor=anchor,prefer=prefer,prior=prior)
        base_by_y={}
        for x,y in baseline:base_by_y.setdefault(y,[]).append(x)
        path_shift=max([min(abs(x-bx) for bx in base_by_y[y])/w for x,y in points if y in base_by_y] or [0])
        baseline_blocked=bool(baseline) and not validate_path(baseline,inset_corridor(free))['valid']
        causal_block=base_coverage>=.55 and coverage<.55
        causal_detour=baseline_blocked and coverage>=.55 and path_shift>.035
        geometry=path_heading(points,(h,w)) if coverage>=.55 and len(points)>4 else None
        validation=validate_path(points,free)
        direction='UNKNOWN';raw='UNKNOWN';status='WAIT';reason='no_continuous_sidewalk';confirmed=False
        if reject_current:reason='camera_motion' if dt>0 else 'nonmonotonic_timestamp';points=[]
        elif coverage<.55 or len(points)<5:
            status='STOP' if causal_block else 'WAIT';reason='detector_blocks_corridor' if causal_block else reason;points=[]
        elif not validation['valid']:reason='path_validation_failed';points=[]
        else:
            heading=geometry['heading_deg'];raw='LEFT' if heading<-18 else 'RIGHT' if heading>18 else 'FORWARD'
            if self.temporal and self.last_direction=='LEFT' and heading<-10:raw='LEFT'
            if self.temporal and self.last_direction=='RIGHT' and heading>10:raw='RIGHT'
            if not self.temporal or self.last_direction is None or raw==self.last_direction:
                direction=raw;confirmed=True;self.pending=None;self.pending_count=0
            else:
                if self.pending!=raw:self.pending=raw;self.pending_since=time_s;self.pending_count=1
                else:self.pending_count+=1
                if self.pending_count>=2 and time_s-self.pending_since>=.15:
                    direction=raw;confirmed=True;self.pending=None;self.pending_count=0
            self.stable=self.stable+1 if confirmed and direction==self.last_direction else 1 if confirmed else 0
            if confirmed:self.last_direction=direction
            status='DETOUR' if causal_detour else 'CANDIDATE'
            reason='direction_confirming' if not confirmed else 'obstacle_detour' if causal_detour else 'current_sidewalk'
            if prefer and geometry['heading_deg']*prefer<-18:
                route_state='conflicts_with_visible_corridor';status='WAIT';direction='UNKNOWN';confirmed=False;reason='route_corridor_conflict';points=[]
        if not points:self.stable=0;self.last_direction=None;self.pending=None;self.pending_count=0
        self.last_path=points;self.last_shape=(h,w)
        text={'LEFT':'候选通道向左延伸','RIGHT':'候选通道向右延伸','FORWARD':'前方有连续人行道候选通道',
              'UNKNOWN':'正在确认前方转向' if points else '暂未找到连续人行道，请先等待'}[direction]
        if status=='DETOUR':text='前方有障碍，'+text
        elif status=='STOP':text='前方通道被障碍占用，请先停下等待'
        elif reason=='camera_motion':text='镜头移动较大，正在重新观察'
        elif reason=='route_corridor_conflict':text='眼前通道与路线方向不一致，请先等待确认路线'
        mode='sidewalk';scan=None
        if reason=='no_continuous_sidewalk':
            from .forward_obstacles import forward_scan
            mode='free_forward';scan=forward_scan(perception,self.forward_half_angle_deg,self.horizontal_fov_deg)
            status='STOP' if scan['obstacles'] else 'FREE'
            reason='forward_obstacle' if scan['obstacles'] else 'free_forward'
            text='正前方检测到障碍，请留意' if scan['obstacles'] else '自由前进模式，持续观察正前方障碍'
        return dict(status=status,direction=direction,reason=reason,text=text,mode=mode,forward_scan=scan,
                    path=[[float(x/w),float(y/h)] for x,y in points],ground_fraction=float(ground.mean()),filtered_fraction=float(free.mean()),
                    path_coverage=float(coverage),baseline_coverage=float(base_coverage),detector_path_shift=float(path_shift),
                    anchor=[float(anchor/w),float(int(h*.94)/h)] if anchor is not None else None,
                    baseline_path=[[float(x/w),float(y/h)] for x,y in baseline],
                    motion=estimate['translation_fraction'],motion_estimate=estimate,stable_frames=self.stable,
                    direction_raw=raw,direction_confirmed=confirmed,path_geometry=geometry,
                    path_validation=validate_path(points,free),route_hint_status=route_state,
                    coordinate_space='image_normalized',spatial_registration='unavailable',
                    detections=detections,_ground=ground,_free=free,_dense_path=points)


def draw(bgr,result):
    frame=bgr.copy();h,w=frame.shape[:2];scale=max(1,w/960)
    ground=cv2.resize(result['_ground'].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    free=cv2.resize(result['_free'].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    frame[ground]=(frame[ground]*.65+np.array([40,165,40])*.35).astype(np.uint8)
    removed=ground&~free;frame[removed]=(frame[removed]*.4+np.array([20,120,255])*.6).astype(np.uint8)
    for d in result['detections']:
        x1,y1,x2,y2=d['box'];a,b,c,e=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
        cv2.rectangle(frame,(a,b),(c,e),(40,170,255),max(2,round(2*scale)))
        cv2.putText(frame,f"{d['label']} {d['score']:.2f}",(a,max(round(18*scale),b-4)),0,.5*scale,(0,220,255),max(1,round(scale)))
    baseline=np.array([[round(x*w),round(y*h)] for x,y in result.get('baseline_path',[])],np.int32)
    if len(baseline)>1:cv2.polylines(frame,[baseline],False,(200,100,200),max(2,round(2*scale)))
    pts=np.array([[round(x*w),round(y*h)] for x,y in result['path']],np.int32)
    if len(pts)>1:
        cv2.polylines(frame,[pts],False,(255,180,0),max(3,round(3*scale)))
        cv2.arrowedLine(frame,tuple(pts[max(0,len(pts)-15)]),tuple(pts[-1]),(255,230,20),max(2,round(2*scale)),tipLength=.4)
    if result.get('forward_scan'):
        scan=result['forward_scan'];x1,y1,x2,y2=scan['sector_box']
        color=(30,110,255) if scan['obstacles'] else (255,205,40)
        cv2.rectangle(frame,(round(x1*w),round(y1*h)),(round(x2*w),min(h-1,round(y2*h))),color,2)
        cv2.putText(frame,f"FORWARD SCAN +/-{scan['half_angle_deg']:g} deg (assumed FOV)",(round(x1*w),round(y1*h)-8),0,.45*scale,color,1)
    cv2.rectangle(frame,(0,0),(w,round(79*scale)),(12,16,20),-1)
    cv2.putText(frame,f"{result['status']} / {result['direction']} | {result['reason']}",(round(10*scale),round(24*scale)),0,.6*scale,(255,255,255),max(1,round(scale)))
    cv2.putText(frame,'Purple: same-start baseline | Blue: current connected candidate',(round(10*scale),round(47*scale)),0,.46*scale,(200,200,200),max(1,round(scale)))
    heading=(result.get('path_geometry') or {}).get('heading_deg');label='heading: --' if heading is None else f'image heading: {heading:+.1f} deg'
    cv2.putText(frame,label+' | image coordinates, no metric clearance',(round(10*scale),round(68*scale)),0,.45*scale,(190,205,220),max(1,round(scale)))
    return frame
