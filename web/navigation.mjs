// Browser perception geometry. Coordinates are normalized image coordinates, not metres.
import {analyzeFloor} from './corridor.mjs';
export function iou(a,b){const w=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return w*h/Math.max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h);}
export function decodeDetector(data,dims,rect,labels){
 const count=dims[2],channels=dims[1],candidates=[];
 for(let i=0;i<count;i++){
  let score=0,cid=0;for(let c=4;c<channels;c++)if(data[c*count+i]>score){score=data[c*count+i];cid=c-4;}
  if(score<.35)continue;
  const cx=data[i],cy=data[count+i],w=data[2*count+i],h=data[3*count+i];
  const box=[((cx-w/2)/640-rect.x)/rect.width,((cy-h/2)/384-rect.y)/rect.height,((cx+w/2)/640-rect.x)/rect.width,((cy+h/2)/384-rect.y)/rect.height].map(x=>Math.max(0,Math.min(1,x)));
  if(box[2]<=box[0]||box[3]<=box[1])continue;
  candidates.push({box,score,class_id:cid,label:labels[cid]||String(cid),blocking:![9,11].includes(cid)});
 }
 const kept=[];for(const d of candidates.sort((a,b)=>b.score-a.score))if(!kept.some(k=>k.class_id===d.class_id&&iou(k.box,d.box)>.55)){kept.push(d);if(kept.length>=50)break;}
 return kept;
}
export function sectorBox(halfAngle=15,fov=60){const span=Math.tan(halfAngle*Math.PI/180)/Math.tan(fov*Math.PI/360);return [Math.max(0,.5-span/2),.55,Math.min(1,.5+span/2),1];}
export function forwardObstacles(detections,halfAngle=15){
 const sector=sectorBox(halfAngle),area=(sector[2]-sector[0])*(sector[3]-sector[1]);
 return {sector,obstacles:detections.filter(d=>d.blocking&&Math.max(0,Math.min(d.box[2],sector[2])-Math.max(d.box[0],sector[0]))*Math.max(0,Math.min(d.box[3],sector[3])-Math.max(d.box[1],sector[1]))/area>=.01)};
}
export function avoidDetections(frame,baseline,detections,threshold){
 const probability=frame.probability.slice(),classes=frame.classes.slice(),{width,height}=frame;
 for(const d of detections.filter(d=>d.blocking)){
  // Full predicted box is conservatively excluded, without claiming physical footprint.
  const [a,b,c,e]=d.box;
  for(let y=Math.max(0,Math.floor(b*height));y<Math.min(height,Math.ceil(e*height));y++)for(let x=Math.max(0,Math.floor(a*width));x<Math.min(width,Math.ceil(c*width));x++){probability[y*width+x]=0;classes[y*width+x]=0;}
 }
 return analyzeFloor(probability,classes,width,height,threshold);
}
export function localNavigation(result,baseline,detections,halfAngle=15){
 const scan=forwardObstacles(detections,halfAngle);
 if(scan.obstacles.length)return {mode:result.target||baseline.target?'corridor':'free_forward',status:'STOP',direction:'UNKNOWN',scan,text:'正前方检测到障碍'};
 if(result.target)return {mode:'corridor',status:'CANDIDATE',direction:result.direction,scan,text:({LEFT:'候选通道向左延伸',RIGHT:'候选通道向右延伸',FORWARD:'前方有连续候选通道'})[result.direction]};
 if(baseline.target&&scan.obstacles.length)return {mode:'corridor',status:'STOP',direction:'UNKNOWN',scan,text:'前方候选通道被障碍占用'};
 return {mode:'free_forward',status:scan.obstacles.length?'STOP':'FREE',direction:'UNKNOWN',scan,text:scan.obstacles.length?'正前方检测到障碍，请留意':'自由前进模式'};
}
export class GuidanceAnnouncements{
 constructor(){this.reset();}
 reset(){this.mode=null;this.status=null;this.direction=null;this.lastAt=-Infinity;}
 update(nav,now){
  const entering=nav.mode==='free_forward'&&this.mode!==nav.mode;
  let text=null;
  if(entering)text='进入自由前进模式，持续观察正前方障碍'+(nav.status==='STOP'?'。'+nav.text:'');
  else if(nav.mode==='free_forward'){if(nav.status==='STOP'&&this.status!=='STOP')text=nav.text;}
  else if(nav.status==='STOP'&&this.status!=='STOP')text=nav.text;
  else if(nav.mode==='corridor'&&(nav.direction!==this.direction||this.mode!==nav.mode)&&now-this.lastAt>=4)text=nav.text;
  this.mode=nav.mode;this.status=nav.status;this.direction=nav.direction;if(text)this.lastAt=now;
  return text;
 }
}

export function semanticObstacles(classes,confidence,width,height,halfAngle=15){
 const [left,top,right]=sectorBox(halfAngle), seen=new Uint8Array(width*height), results=[];
 const fixed=new Set([0,1,4,8,12,17,32,33]);
 const inside=i=>{const x=i%width,y=Math.floor(i/width);return x>=left*width&&x<right*width&&y>=top*height&&fixed.has(classes[i])&&confidence[i]>=.6;};
 const minimum=Math.max(8,(right-left)*(1-top)*width*height*.01);
 for(let i=0;i<seen.length;i++){
  if(seen[i]||!inside(i))continue;seen[i]=1;const queue=[i];let minX=width,maxX=0,minY=height,maxY=0;
  for(let j=0;j<queue.length;j++){const v=queue[j],x=v%width,y=Math.floor(v/width);minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
   for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const p=ny*width+nx;if(!seen[p]&&inside(p)){seen[p]=1;queue.push(p);}}
  }
  if(queue.length>=minimum)results.push({box:[minX/width,minY/height,(maxX+1)/width,(maxY+1)/height],blocking:true,label:'semantic obstacle',source:'semantic',score:1});
 }
 return results;
}
