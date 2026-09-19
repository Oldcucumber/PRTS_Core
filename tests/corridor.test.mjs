import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeFloor} from '../web/corridor.mjs';
const w=48,h=80;
function scene(predicate){const p=new Float32Array(w*h),c=new Float32Array(w*h);for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(predicate(x,y)){p[y*w+x]=.9;c[y*w+x]=3;}return [p,c];}
test('unknown when no near-field floor exists',()=>{
  const [p,c]=scene((x,y)=>y<50);
  const r=analyzeFloor(p,c,w,h);
  assert.equal(r.direction,'UNKNOWN');assert.equal(r.fraction,0);
});
test('straight corridor and confidence threshold',()=>{
  const [p,c]=scene((x,y)=>x>=12&&x<36);
  assert.equal(analyzeFloor(p,c,w,h).direction,'FORWARD');
  assert.equal(analyzeFloor(p,c,w,h,.95).direction,'UNKNOWN');
});
test('obstacle across corridor stops path below obstacle',()=>{
  const [p,c]=scene((x,y)=>x>=12&&x<36&&!(y>=40&&y<47));
  const r=analyzeFloor(p,c,w,h);
  assert.ok(r.points.length>0);assert.ok(r.points.every(([,y])=>y>47));
  assert.equal(r.region[30*w+24],0);
});
test('side corridor produces left and right estimates',()=>{
  for(const [left,right,direction] of [[3,23,'LEFT'],[26,46,'RIGHT']]){
    const [p,c]=scene(x=>x>=left&&x<right);
    assert.equal(analyzeFloor(p,c,w,h).direction,direction);
  }
});
