# Bundled third-party assets

- ONNX Runtime Web 1.30.0: https://github.com/microsoft/onnxruntime — MIT; see `licenses/onnxruntime-MIT.txt`. Runtime copyright headers are preserved in the bundled files.
- SegFormer-B0 / ADE20K weights: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 — exported to ONNX without training changes. The model repository labels its license `other` and points to NVIDIA's SegFormer project. Its upstream license limits use to non-commercial research/evaluation; see the complete `licenses/segformer-NVIDIA.txt` and the preserved `licenses/segformer-model-card.md`. No MIT license is asserted for the model weights. Original project: https://github.com/NVlabs/SegFormer.
- The test video is user-provided project data, included as `test-video.mp4` to preserve the test input.
- Depth Anything V2 Small: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf — Apache-2.0. Exported to fixed 252×252 ONNX with the same pretrained weights, no additional training. See `licenses/depth-anything-v2-Apache-2.0.txt` and `licenses/depth-anything-v2-model-card.md`. Upstream: https://github.com/DepthAnything/Depth-Anything-V2.

Runtime, models and video are served from this site's own directory. These source links are attribution only, not runtime dependencies.
