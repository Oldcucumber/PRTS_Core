import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeFloor} from '../web/corridor.mjs';
import {alignDepth,fuseDepth} from '../web/depth-fusion.mjs';
function fixture(){
  const width=48,height=80,probability=new Float32Array(width*height),classes=new Float32Array(width*height),depth=new Float32Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x;depth[i]=.2+.6*y/height+.04*x/width;
    if(y>20&&x>5&&x<43){classes[i]=3;probability[i]=.9;}
  }
  const frame={width,height,probability,classes};return {frame,depth,baseline:analyzeFloor(probability,classes,width,height)};
}
test('planar floor remains unchanged',()=>{
  const {frame,depth,baseline}=fixture(),fused=fuseDepth(frame,baseline,depth);
  assert.equal(fused.fit.valid,true);assert.equal(fused.removedFraction,0);
  assert.deepEqual(fused.result.region,baseline.region);
});
test('closer inconsistent patch is excluded, without adding any new floor',()=>{
  const {frame,depth,baseline}=fixture();
  for(let y=42;y<54;y++)for(let x=18;x<30;x++)depth[y*frame.width+x]+=.45;
  const fused=fuseDepth(frame,baseline,depth);
  assert.equal(fused.fit.valid,true);assert.ok(fused.removedFraction>.03);
  assert.equal(fused.excluded[48*48+24],1);assert.equal(fused.result.region[48*48+24],0);
  assert.ok(fused.result.region.every((v,i)=>!v||baseline.region[i]));
});
test('constant and invalid depth fall back to the segmentation result',()=>{
  const {frame,baseline}=fixture();
  for(const value of [1,NaN]){
    const depth=new Float32Array(frame.width*frame.height).fill(value),r=fuseDepth(frame,baseline,depth);
    assert.equal(r.fit.valid,false);assert.deepEqual(r.result.region,baseline.region);
  }
});

test('removing near seeds cannot select an unrelated floor component',()=>{
  const {frame,depth}=fixture(),{width,height}=frame;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x;
    frame.classes[i]=y>20&&((x>2&&x<32)||(x>35&&x<47))?3:0;
    frame.probability[i]=frame.classes[i]===3?.9:0;
    if(y>=66&&x<32)depth[i]+=.6;
  }
  const baseline=analyzeFloor(frame.probability,frame.classes,width,height);
  const fused=fuseDepth(frame,baseline,depth);
  assert.equal(fused.fit.valid,true);assert.ok(fused.removedFraction>.1);
  assert.ok(fused.result.region.every((value,i)=>!value||baseline.region[i]));
  assert.equal(fused.result.fraction,0);
});
test('depth padding is removed while preserving image-coordinate alignment',()=>{
  const values=new Float32Array(8*8).fill(-100);
  for(let y=0;y<8;y++)for(let x=2;x<6;x++)values[y*8+x]=y*10+x;
  const aligned=alignDepth(values,8,8,{x:.25,y:0,width:.5,height:1},4,8);
  assert.equal(aligned[0],2);assert.equal(aligned[3],5);assert.equal(aligned[31],75);
});
