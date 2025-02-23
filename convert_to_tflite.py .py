import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("sin_predictor_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
with open("sin_predictor_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Model successfully converted to TFLite format.")
