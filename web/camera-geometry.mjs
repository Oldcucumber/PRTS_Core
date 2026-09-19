// Fit the complete source into a target rectangle; never discard source pixels.
export function containRect(sourceWidth,sourceHeight,targetWidth,targetHeight){
  if([sourceWidth,sourceHeight,targetWidth,targetHeight].some(n=>!Number.isFinite(n)||n<=0))return null;
  const scale=Math.min(targetWidth/sourceWidth,targetHeight/sourceHeight);
  const width=Math.max(1,Math.round(sourceWidth*scale));
  const height=Math.max(1,Math.round(sourceHeight*scale));
  return {x:Math.floor((targetWidth-width)/2),y:Math.floor((targetHeight-height)/2),width,height};
}

// Remove only synthetic letterbox padding from the model's output grid.
// The returned grid still covers the entire camera image.
export function unpadPrediction(probability,classes,width,height,content){
  const outWidth=Math.max(1,Math.round(content.width*width));
  const outHeight=Math.max(1,Math.round(content.height*height));
  const outputProbability=new Float32Array(outWidth*outHeight);
  const outputClasses=new Float32Array(outWidth*outHeight);
  for(let y=0;y<outHeight;y++)for(let x=0;x<outWidth;x++){
    const sx=Math.min(width-1,Math.max(0,Math.floor((content.x+(x+.5)*content.width/outWidth)*width)));
    const sy=Math.min(height-1,Math.max(0,Math.floor((content.y+(y+.5)*content.height/outHeight)*height)));
    const input=sy*width+sx,output=y*outWidth+x;
    outputProbability[output]=probability[input];outputClasses[output]=classes[input];
  }
  return {probability:outputProbability,classes:outputClasses,width:outWidth,height:outHeight};
}

export function cameraConstraints(deviceId=''){
  return {
    ...(deviceId?{deviceId:{exact:deviceId}}:{facingMode:{ideal:'environment'}}),
    // Resolution preferences do not dictate a viewport-shaped aspect ratio.
    width:{ideal:1280},height:{ideal:1280},resizeMode:{ideal:'none'},
  };
}
