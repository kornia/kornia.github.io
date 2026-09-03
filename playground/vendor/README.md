`onnx.proto` is the ONNX protobuf schema (proto2), copied verbatim from the `onnx` Python package
(`onnx/onnx.proto`, Apache-2.0). `pipeline.js` parses it at runtime with protobuf.js to decode the
exported graphs, splice them into a pipeline and re-encode the result in the browser. Keep it in
step with the `onnx` version recorded in `registry.json`.
