import test from 'node:test';
import assert from 'node:assert/strict';
import {containRect,unpadPrediction,cameraConstraints} from '../web/camera-geometry.mjs';

test('landscape image is scaled into portrait model without removing edges',()=>{
  assert.deepEqual(containRect(1280,720,192,320),{x:0,y:106,width:192,height:108});
});
test('portrait image fits landscape target without stretching',()=>{
  const fit=containRect(720,1280,844,390);
  assert.equal(fit.height,390);assert.ok(Math.abs(fit.width/fit.height-720/1280)<.002);
  assert.ok(fit.x>0);assert.equal(fit.y,0);
});
test('matching aspect does not pad; missing dimensions cannot produce a frame',()=>{
  assert.deepEqual(containRect(720,1280,360,640),{x:0,y:0,width:360,height:640});
  assert.equal(containRect(0,0,390,844),null);
});
test('synthetic floor predictions in padding never reach navigation',()=>{
  const p=new Float32Array(48*80).fill(1),c=new Float32Array(48*80).fill(3);
  for(let y=26;y<=53;y++)for(let x=0;x<48;x++){p[y*48+x]=.1;c[y*48+x]=0;}
  // Content spans output coordinates 26.5..53.5. Exclude wholly padded rows.
  const r=unpadPrediction(p,c,48,80,{x:0,y:106/320,width:1,height:108/320});
  assert.equal(r.width,48);assert.equal(r.height,27);
  assert.ok(r.classes.every(v=>v===0));assert.ok(r.probability.every(v=>v<.2));
});
test('all four content borders survive padding removal',()=>{
  const p=new Float32Array(8*8),c=new Float32Array(8*8);
  // Full source region has four colored corners, while rows outside are padding.
  for(let y=2;y<6;y++)for(let x=0;x<8;x++){p[y*8+x]=.8;c[y*8+x]=y*10+x;}
  const r=unpadPrediction(p,c,8,8,{x:0,y:.25,width:1,height:.5});
  assert.deepEqual([r.classes[0],r.classes[7],r.classes[24],r.classes[31]],[20,27,50,57]);
});
test('camera request avoids viewport crop and preserves exact device selection',()=>{
  const auto=cameraConstraints(),explicit=cameraConstraints('front-or-wide-camera');
  assert.equal(auto.resizeMode.ideal,'none');assert.equal('aspectRatio' in auto,false);
  assert.equal(auto.facingMode.ideal,'environment');
  assert.deepEqual(explicit.deviceId,{exact:'front-or-wide-camera'});
  assert.equal('facingMode' in explicit,false);assert.equal('deviceId' in auto,false);
});
