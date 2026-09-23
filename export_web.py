"""Export the existing SegFormer-B0 to a small, fixed-shape browser model."""
import json
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import SegformerForSemanticSegmentation

MODEL = 'nvidia/segformer-b0-finetuned-ade-512-512'


class FloorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()

    def forward(self, pixel_values):
        logits = self.model(pixel_values=pixel_values).logits
        # Return only floor probability and winning class, reducing GPU readback.
        probabilities = logits.softmax(1)
        return probabilities[:, 3], logits.argmax(1).to(torch.float32), probabilities.max(1).values


def main():
    torch.set_num_threads(4)
    target = Path('web/models')
    target.mkdir(parents=True, exist_ok=True)
    model = FloorModel().eval()
    report = {}
    for name, height, width in [('fast', 320, 192), ('quality', 512, 288)]:
        cap = cv2.VideoCapture('VID20260919182406.mp4')
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError('Cannot read test video')
        rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32)/255 - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        sample = torch.from_numpy(normalized.transpose(2,0,1).copy()[None])
        path = target / f'floor-{name}.onnx'
        torch.onnx.export(model, sample, str(path), input_names=['pixel_values'],
                          output_names=['floor_probability', 'winning_class', 'semantic_confidence'],
                          opset_version=17, dynamo=False)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        actual = session.run(None, {'pixel_values':sample.numpy()})
        with torch.inference_mode():
            expected = [x.numpy() for x in model(sample)]
        error = float(np.abs(actual[0]-expected[0]).max())
        agreement = float((actual[1]==expected[1]).mean())
        assert error < 1e-3 and agreement > .999, (error,agreement)
        report[name] = {'height':height, 'width':width, 'bytes':path.stat().st_size,
                        'max_probability_error':error, 'class_agreement':agreement}
        print(name, report[name], flush=True)
    (target/'export-validation.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
