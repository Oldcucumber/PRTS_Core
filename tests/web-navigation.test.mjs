import test from 'node:test';
import assert from 'node:assert/strict';
import {GuidanceAnnouncements,localNavigation,forwardObstacles,decodeDetector,avoidDetections,semanticObstacles} from '../web/navigation.mjs';
import {analyzeFloor} from '../web/corridor.mjs';
test('free mode speaks once, alerts on a new forward obstacle and rearms',()=>{
 const g=new GuidanceAnnouncements(),empty={target:null,direction:'UNKNOWN'},free=localNavigation(empty,empty,[]);
 assert.equal(free.mode,'free_forward');assert.ok(g.update(free,0));for(let i=1;i<30;i++)assert.equal(g.update(free,i),null);
 const blocked=localNavigation(empty,empty,[{blocking:true,box:[.4,.6,.6,.9]}]);assert.equal(blocked.status,'STOP');assert.ok(g.update(blocked,31));assert.equal(g.update(blocked,35),null);assert.equal(g.update(free,36),null);assert.ok(g.update(blocked,37));
});
test('side, overhead and nonblocking signs do not trigger the forward sector',()=>{
 const result=forwardObstacles([{blocking:true,box:[0,.6,.15,1]},{blocking:true,box:[.4,0,.6,.3]},{blocking:false,box:[.3,.6,.7,.9]}]);assert.equal(result.obstacles.length,0);
});
test('detector crops undo letterboxing and class-aware NMS removes duplicates',()=>{
 const raw=new Float32Array(84*2);for(let i=0;i<2;i++){raw[i]=320;raw[2+i]=192;raw[4+i]=128;raw[6+i]=100;raw[8+i]=.9;}
 const out=decodeDetector(raw,[1,84,2],{x:0,y:0,width:1,height:1},{0:'person'});assert.equal(out.length,1);assert.ok(Math.abs(out[0].box[0]-.4)<1e-6);assert.equal(out[0].label,'person');
});
test('detector exclusion never creates a traversable pixel',()=>{
 const width=40,height=60,frame={width,height,probability:new Float32Array(width*height).fill(.9),classes:new Float32Array(width*height).fill(3)};
 const base=analyzeFloor(frame.probability,frame.classes,width,height);const after=avoidDetections(frame,base,[{blocking:true,box:[0,.7,1,1]}],.55);assert.equal(after.target,null);assert.ok(after.fraction<=base.fraction);
});
test('semantic wall in forward sector is detected, low-confidence pixels are excluded',()=>{
 const c=new Float32Array(40*60).fill(3),p=new Float32Array(c.length).fill(.95);for(let y=40;y<60;y++)for(let x=17;x<23;x++)c[y*40+x]=0;
 assert.ok(semanticObstacles(c,p,40,60).length);p.fill(.2);assert.equal(semanticObstacles(c,p,40,60).length,0);
});
