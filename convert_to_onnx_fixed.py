import tensorflow as tf
import tf2onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load Keras model
model = tf.keras.models.load_model("models/cnn_ecg.h5", compile=False)

# Define input signature
spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("cnn_ecg.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Saved cnn_ecg.onnx")

# Quantize dynamically
quantize_dynamic("cnn_ecg.onnx", "cnn_ecg_quant.onnx", weight_type=QuantType.QInt8)
print("Saved cnn_ecg_quant.onnx")