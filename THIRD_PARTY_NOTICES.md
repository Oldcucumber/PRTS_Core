# Bundled third-party assets

- ONNX Runtime Web 1.30.0: https://github.com/microsoft/onnxruntime — MIT; see `licenses/onnxruntime-MIT.txt`. Runtime copyright headers are preserved in the bundled files.
- SegFormer-B0 / ADE20K weights: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 — exported to ONNX without training changes. The model repository labels its license `other` and points to NVIDIA's SegFormer project. Its upstream license limits use to non-commercial research/evaluation; see the complete `licenses/segformer-NVIDIA.txt` and the preserved `licenses/segformer-model-card.md`. No MIT license is asserted for the model weights. Original project: https://github.com/NVlabs/SegFormer.
- The test video is user-provided project data, included as `test-video.mp4` to preserve the test input.
- Depth Anything V2 Small: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf — Apache-2.0. Exported to fixed 252×252 ONNX with the same pretrained weights, no additional training. See `licenses/depth-anything-v2-Apache-2.0.txt` and `licenses/depth-anything-v2-model-card.md`. Upstream: https://github.com/DepthAnything/Depth-Anything-V2.

Runtime and models are served from this site's own directory. These source links are attribution only, not runtime dependencies.

- YOLO11n: Ultralytics pretrained COCO detector, exported to 384×640 ONNX without additional training. Source https://github.com/ultralytics/ultralytics — AGPL-3.0; see `licenses/Ultralytics-AGPL-3.0.txt` in the deployed site (`licenses/stage2/Ultralytics-AGPL-3.0.txt` in the repository). Metadata: `models/detector-manifest.json`. PRTS source: https://github.com/Oldcucumber/PRTS_Core.
- SegFormer export also exposes winning-class confidence without retraining. Floor, sidewalk and path are candidate walking classes; road is excluded.

- Local speech: sherpa-onnx v1.13.2 WASM (Apache-2.0), SenseVoice Small (upstream license preserved), Silero VAD (MIT). Official release URL, byte counts and SHA256 hashes are recorded in speech/manifest.json; original license texts are bundled in speech/. The upstream .data is split into four byte-preserving parts to stay below Git file-size limits, then joined in the speech worker. No cloud ASR service is used.
