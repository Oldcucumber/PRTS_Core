"""Separate-process CPU samples for numerical/INT8 quality and memory comparison."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,psutil
from prts_core.onnx_perception import ONNXPerception
from perception_metrics import label_metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--tag',required=True)
    ap.add_argument('--reference');ap.add_argument('--low-memory',action='store_true');ap.add_argument('--data',type=Path,default=Path('outputs/stage2/perception/data'))
    ap.add_argument('--provider',default='CPUExecutionProvider')
    a=ap.parse_args();out=Path('outputs/stage2/perception/cpu-validation')/a.tag;out.mkdir(parents=True,exist_ok=True)
    process=psutil.Process();baseline=process.memory_info().rss;start=time.perf_counter()
    m=ONNXPerception(a.model,'outputs/stage2/perception/onnx/yolo11n-rect',low_memory=a.low_memory,providers=[a.provider]);load_s=time.perf_counter()-start;rows=[]
    for manifest in sorted(a.data.glob('dev-*/manifest.json')):
        data=json.loads(manifest.read_text());chest=data['frames'];selected=[chest[len(chest)//2],*data['human_label_frames']]
        for record in selected:
            image=manifest.parent/record['file'];p=m(cv2.imread(str(image)))
            key=data['case']+'-'+image.parent.name+'-'+image.stem
            np.savez_compressed(out/(key+'.npz'),classes=p['classes'],sidewalk=p['sidewalk'],sidewalk_class=p['sidewalk_class'])
            row={'key':key,'image':str(image),'timings':p['timings'],'detections':p['detections']}
            if record.get('label'):row['label_metrics']=label_metrics(p,manifest.parent/record['label'])
            if a.reference:
                old=dict(np.load(Path('outputs/stage2/perception/cpu-validation')/a.reference/(key+'.npz')))
                row['class_agreement']=float((p['classes']==old['classes']).mean())
                pw=p['sidewalk_class']&(p['sidewalk']>=.5);ow=old['sidewalk_class']&(old['sidewalk']>=.5)
                row['walkable_iou_vs_fp32']=float((pw&ow).sum()/max(1,(pw|ow).sum()))
            rows.append(row);print(key,p['timings'],row.get('class_agreement'),flush=True)
    info=process.memory_info();result={'model':a.model,'low_memory':a.low_memory,'rows':rows,'load_s':load_s,'baseline_rss':baseline,
        'final_rss':info.rss,'peak_working_set':getattr(info,'peak_wset',None),'torch_imported':'torch' in sys.modules,
        'provider':a.provider,'session_providers':m.seg.get_providers(),'quantization_acceptance_predeclared':{
        'mean_class_agreement_at_least':.98,'minimum_walkable_iou_vs_fp32':.95,'human_walkable_iou_drop_at_most':.01}}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
