import onnx
import onnxruntime as ort
import numpy as np

# Load and verify model
onnx_model = onnx.load("tinycnn.onnx")
onnx.checker.check_model(onnx_model)

# Create inference session
ort_session = ort.InferenceSession("tinycnn.onnx", providers=["CPUExecutionProvider"])

# Create a dummy input (1 image, 3x32x32)
dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {"input": dummy})
print("Output shape:", outputs[0].shape)
print("Predicted logits:", outputs[0])
print("Predicted class:", outputs[0].argmax(axis=1))
