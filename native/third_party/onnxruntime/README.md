The unmodified `onnxruntime_c_api.h` and MIT license are from Microsoft's
ONNX Runtime tag `v1.23.2`:

- https://github.com/microsoft/onnxruntime/tree/v1.23.2/include/onnxruntime/core/session
- Header SHA256: `71125e66180a991d65c9bdbad4aa20daaa1f7a48c7a5c0fa5f18f250ac839a02`

The bridge requests API version 23 from a runtime supplied by the application.
It does not embed an ONNX Runtime library. Keep the application's single runtime
compatible with its other model dependencies.
