// Pure navigation state machine. All times are monotonic milliseconds.
export function controlIntent(text){
 const s=text.replace(/[\s，。！？,.!?]/g,'');
 return ({'暂停导航':'pause','停止导航':'pause','继续导航':'resume','开始导航':'resume','静音':'mute','恢复播报':'unmute'})[s]||null;
}
export class NavigationSession {
 constructor(emit,clock=()=>performance.now()){this.emit=emit;this.clock=clock;this.sequence=0;this.session=0;this.active=false;}
 event(type,fields={}){if(type==='state'){const key=JSON.stringify(fields);if(this.lastState===key)return;this.lastState=key;}const event={type,session:this.session,sequence:++this.sequence,emitted_ms:this.clock(),...fields};this.emit(event);return event;}
 start(){this.lastState=null;this.session++;this.active=true;this.paused=false;this.reset();this.event('state',{state:'initializing'});}
 reset(){this.mode=null;this.pending=null;this.pendingSince=0;this.direction=null;this.lastDirectionAt=-Infinity;this.blocked=false;this.clearSince=null;this.lastFresh=this.clock();this.interrupted=false;this.seen=false;}
 ready(){this.lastFresh=this.clock();this.seen=true;}
 say(text,priority,observation=this.clock(),group='navigation'){this.event('speech_request',{text,priority,replace_group:group,observed_ms:observation,expires_ms:observation+2000});}
 stop(){if(!this.active)return;this.active=false;this.event('speech_cancel');this.event('state',{state:'ended'});}
 pause(){if(!this.active||this.paused)return;this.paused=true;this.event('speech_cancel');this.event('state',{state:'paused'});}
 resume(){if(!this.active||!this.paused)return;this.paused=false;this.reset();this.event('state',{state:'initializing'});}
 tick(){if(!this.active||this.paused||!this.seen||this.interrupted)return;if(this.clock()-this.lastFresh>2000){this.interrupted=true;this.mode=null;this.pending=null;this.event('speech_cancel');this.event('state',{state:'interrupted'});this.say('感知中断，请暂停前进',90);}}
 observe(scene){
 if(!this.active||this.paused)return;
 const now=this.clock(),observed=scene.observedAt??now-scene.ageMs;
 if(now-observed>2000){this.tick();return;}
 this.lastFresh=observed;this.seen=true;
 if(this.interrupted){this.reset();this.lastFresh=observed;this.seen=true;}
 const nav=scene.nav;this.event('scene',{observed_ms:observed,mode:nav.mode,direction:nav.direction,obstacles:nav.scan.obstacles.length});
 const hazard=nav.scan.obstacles.length>0||nav.status==='STOP';
 if(hazard){this.clearSince=null;if(!this.blocked){this.blocked=true;this.event('state',{state:'blocked'});this.say('正前方有障碍，请暂停前进',100,observed);}return;}
 if(this.blocked){this.clearSince??=now;if(now-this.clearSince<1000)return;this.blocked=false;this.direction=null;}
 const key=nav.mode==='free_forward'?'free_forward':'corridor:'+nav.direction;
 if(this.pending!==key){this.pending=key;this.pendingSince=now;}
 if(now-this.pendingSince<(nav.mode==='free_forward'?1000:500))return;
 if(nav.mode==='free_forward'){
 if(this.mode!=='free_forward'){this.mode='free_forward';this.direction=null;this.event('state',{state:'free_forward'});this.say('进入自由前进模式',40,observed);}return;
 }
 const entered=this.mode!=='corridor';this.mode='corridor';this.event('state',{state:'corridor',direction:nav.direction});
 if((entered||this.direction!==nav.direction)&&now-this.lastDirectionAt>=4000){this.direction=nav.direction;this.lastDirectionAt=now;this.say(({LEFT:'向左调整',RIGHT:'向右调整',FORWARD:'沿当前方向前进'})[nav.direction]||'请观察前方',40,observed);}
 }
}
