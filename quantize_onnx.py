# quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model-onnx/model.onnx", "model-onnx/model_int8.onnx", weight_type=QuantType.QInt8)
print("Saved quantized ONNX -> model-onnx/model_int8.onnx")

