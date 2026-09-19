// Match CSS object-fit: cover without rotating or stretching the sensor image.
export function coverCrop(sourceWidth,sourceHeight,viewWidth,viewHeight){
  if([sourceWidth,sourceHeight,viewWidth,viewHeight].some(n=>!Number.isFinite(n)||n<=0))return null;
  const aspect=viewWidth/viewHeight;
  const width=Math.min(sourceWidth,sourceHeight*aspect);
  const height=width/aspect;
  return {x:(sourceWidth-width)/2,y:(sourceHeight-height)/2,width,height};
}

export function cameraConstraints(viewWidth,viewHeight){
  const aspect=viewWidth>0&&viewHeight>0?viewWidth/viewHeight:9/16;
  const scale=1280/Math.max(aspect,1);
  return {
    facingMode:{ideal:'environment'},
    width:{ideal:Math.round(aspect*scale)},
    height:{ideal:Math.round(scale)},
    aspectRatio:{ideal:aspect},
    resizeMode:{ideal:'crop-and-scale'},
  };
}
