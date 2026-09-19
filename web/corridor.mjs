// Image-space geometry only. No metric distance or body clearance is implied.
export function analyzeFloor(probability, classes, width, height, threshold=.55) {
  const n = width*height, mask = new Uint8Array(n);
  for (let i=0;i<n;i++) mask[i] = probability[i] >= threshold && classes[i] === 3 ? 1 : 0;
  const labels = new Int32Array(n), queue = new Int32Array(n);
  let id=0, best=0, bestSeeds=0;
  for (let i=0;i<n;i++) {
    if (!mask[i] || labels[i]) continue;
    id++; let head=0,tail=1,seeds=0; queue[0]=i; labels[i]=id;
    while(head<tail) {
      const p=queue[head++], x=p%width, y=Math.floor(p/width);
      if(y>=height*.85 && x>=width*.2 && x<width*.8) seeds++;
      for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
        const nx=x+dx,ny=y+dy;
        if(nx<0 || nx>=width || ny<0 || ny>=height) continue;
        const q=ny*width+nx;
        if(mask[q] && !labels[q]) { labels[q]=id; queue[tail++]=q; }
      }
    }
    if(seeds>bestSeeds) { bestSeeds=seeds; best=id; }
  }
  const region=new Uint8Array(n), inset=new Uint8Array(n);
  let area=0;
  for(let i=0;i<n;i++) { region[i]=best && labels[i]===best ? 1 : 0; area+=region[i]; }
  // One low-resolution pixel margin: a display heuristic, not physical clearance.
  for(let y=1;y<height-1;y++) for(let x=1;x<width-1;x++) {
    let good=1;
    for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) good &= region[(y+dy)*width+x+dx];
    inset[y*width+x]=good;
  }
  const points=[];
  let previous=width/2;
  for(let y=Math.floor(height*.94);y>height*.25;y-=Math.max(1,Math.floor(height/35))) {
    const runs=[];
    for(let x=0;x<width;) {
      if(!inset[y*width+x]) { x++; continue; }
      const start=x;
      while(x<width && inset[y*width+x]) x++;
      if(x-start>=width*.08) runs.push([start,x]);
    }
    if(!runs.length) break;
    runs.sort((a,b)=>Math.abs((a[0]+a[1])/2-previous)-Math.abs((b[0]+b[1])/2-previous));
    const x=Math.floor((runs[0][0]+runs[0][1])/2);
    if(points.length) {
      const [px,py]=points.at(-1), steps=Math.max(Math.abs(px-x),Math.abs(py-y))*2;
      let clear=true;
      for(let s=0;s<=steps;s++) {
        const ix=Math.round(px+(x-px)*s/steps), iy=Math.round(py+(y-py)*s/steps);
        if(!inset[iy*width+ix]) { clear=false; break; }
      }
      if(!clear) break;
    }
    points.push([x,y]); previous=x;
  }
  const target=points.length>=5 ? points[Math.min(points.length-1,Math.max(1,Math.floor(points.length*.7)))] : null;
  const offset=target ? (target[0]-width/2)/(width/2) : null;
  const direction=target ? offset<-.15 ? 'LEFT' : offset>.15 ? 'RIGHT' : 'FORWARD' : 'UNKNOWN';
  return { region, points, target, offset, direction, fraction:area/n, width, height };
}
