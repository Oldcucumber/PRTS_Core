import test from 'node:test';
import assert from 'node:assert/strict';
import {coverCrop,cameraConstraints} from '../web/camera-geometry.mjs';

test('landscape sensor retains upright central portrait field of view',()=>{
  const crop=coverCrop(1280,720,390,844);
  assert.equal(crop.y,0);assert.equal(crop.height,720);
  assert.ok(Math.abs(crop.width/crop.height-390/844)<1e-9);
  assert.equal(crop.x+crop.width/2,640);
  assert.ok(crop.width<1280);
});
test('portrait sensor in landscape viewport crops vertically without rotation',()=>{
  const crop=coverCrop(720,1280,844,390);
  assert.equal(crop.x,0);assert.equal(crop.width,720);
  assert.ok(Math.abs(crop.width/crop.height-844/390)<1e-9);
  assert.equal(crop.y+crop.height/2,640);
});
test('matching aspect does not crop; unavailable dimensions cannot produce a frame',()=>{
  assert.deepEqual(coverCrop(720,1280,360,640),{x:0,y:0,width:720,height:1280});
  assert.equal(coverCrop(0,0,390,844),null);
  assert.equal(coverCrop(720,1280,390,0),null);
});
test('request native portrait or landscape without hard device constraints',()=>{
  const portrait=cameraConstraints(390,844),landscape=cameraConstraints(844,390);
  assert.ok(portrait.width.ideal<portrait.height.ideal);
  assert.ok(landscape.width.ideal>landscape.height.ideal);
  assert.equal(portrait.aspectRatio.ideal,390/844);
  assert.equal(portrait.facingMode.ideal,'environment');
  assert.equal(portrait.resizeMode.ideal,'crop-and-scale');
});
