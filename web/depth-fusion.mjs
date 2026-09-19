import {analyzeFloor} from './corridor.mjs';

export function alignDepth(data,sourceWidth,sourceHeight,content,width,height){
  const aligned=new Float32Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const sx=Math.max(0,Math.min(sourceWidth-1,Math.floor((content.x+(x+.5)*content.width/width)*sourceWidth)));
    const sy=Math.max(0,Math.min(sourceHeight-1,Math.floor((content.y+(y+.5)*content.height/height)*sourceHeight)));
    aligned[y*width+x]=data[sy*sourceWidth+sx];
  }
  return aligned;
}
function quantile(values,q){return values[Math.min(values.length-1,Math.floor((values.length-1)*q))];}
function plane(points,weights){
  const a=[[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  points.forEach(([x,y,z],i)=>{
    const w=weights?weights[i]:1,v=[x,y,1];
    for(let r=0;r<3;r++){for(let c=0;c<3;c++)a[r][c]+=w*v[r]*v[c];a[r][3]+=w*v[r]*z;}
  });
  for(let c=0;c<3;c++){
    let pivot=c;for(let r=c+1;r<3;r++)if(Math.abs(a[r][c])>Math.abs(a[pivot][c]))pivot=r;
    if(Math.abs(a[pivot][c])<1e-9)return null;
    [a[c],a[pivot]]=[a[pivot],a[c]];
    const divisor=a[c][c];for(let j=c;j<4;j++)a[c][j]/=divisor;
    for(let r=0;r<3;r++)if(r!==c){const factor=a[r][c];for(let j=c;j<4;j++)a[r][j]-=factor*a[c][j];}
  }
  return a.map(row=>row[3]);
}
const residual=(p,m)=>Math.abs(p[2]-(m[0]*p[0]+m[1]*p[1]+m[2]));

// Relative inverse depth is approximately affine on a plane in image coordinates.
// This robust fit is an experimental consistency filter, not metric navigation.
export function fuseDepth(frame,baseline,depth,threshold=.55){
  const {width,height}=frame,n=width*height,heat=new Uint8Array(n),excluded=new Uint8Array(n);
  const values=Array.from(depth).filter(Number.isFinite).sort((a,b)=>a-b);
  const unchanged=reason=>({result:baseline,heat,excluded,removedFraction:0,fit:{valid:false,reason}});
  if(values.length<n*.95)return unchanged('深度包含无效值');
  const lo=quantile(values,.05),hi=quantile(values,.95),range=hi-lo;
  if(!Number.isFinite(range)||range<1e-6)return unchanged('深度变化不足');
  const normalized=new Float32Array(n),samples=[];
  for(let i=0;i<n;i++){
    normalized[i]=(depth[i]-lo)/range;
    heat[i]=Math.round(Math.max(0,Math.min(1,normalized[i]))*255);
    if(baseline.region[i]&&Number.isFinite(depth[i]))samples.push([(i%width)/width,Math.floor(i/width)/height,normalized[i]]);
  }
  if(samples.length<30)return unchanged('地面样本不足');
  const ys=samples.map(p=>p[1]);
  if(Math.max(...ys)-Math.min(...ys)<.2)return unchanged('地面纵向覆盖不足');
  // Deterministic RANSAC initialization reduces foreground outlier influence.
  let best=null,bestCount=0,seed=713;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(let k=0;k<64;k++){
    const pick=()=>samples[Math.floor(random()*samples.length)];
    const model=plane([pick(),pick(),pick()]);if(!model)continue;
    const count=samples.reduce((sum,p)=>sum+(residual(p,model)<.08?1:0),0);
    if(count>bestCount){best=model;bestCount=count;}
  }
  if(!best||bestCount/samples.length<.55)return unchanged('未找到稳定地面趋势');
  const inliers=samples.filter(p=>residual(p,best)<.08);
  best=plane(inliers)||best;
  const errors=inliers.map(p=>residual(p,best)).sort((a,b)=>a-b);
  const cutoff=Math.max(.10,quantile(errors,.5)*4);
  const suspects=new Uint8Array(n);
  for(let i=0;i<n;i++)if(baseline.region[i]){
    const predicted=best[0]*(i%width)/width+best[1]*Math.floor(i/width)/height+best[2];
    suspects[i]=Math.abs(normalized[i]-predicted)>cutoff?1:0;
  }
  let removed=0,baseArea=0;
  // Keep the baseline component as the upper bound. Otherwise, removing its
  // near seeds could make analyzeFloor select an unrelated floor component.
  const classes=frame.classes.map((value,i)=>baseline.region[i]?value:0);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x;baseArea+=baseline.region[i];if(!suspects[i])continue;
    let support=0;
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
      const xx=x+dx,yy=y+dy;if(xx>=0&&xx<width&&yy>=0&&yy<height)support+=suspects[yy*width+xx];
    }
    if(support>=3){excluded[i]=1;classes[i]=0;removed++;}
  }
  return {result:analyzeFloor(frame.probability,classes,width,height,threshold),heat,excluded,
    removedFraction:baseArea?removed/baseArea:0,
    fit:{valid:true,inlierFraction:bestCount/samples.length,cutoff,coefficients:best}};
}
