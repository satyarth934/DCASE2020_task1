#### For quantizing the keras model to TF lite model
import sys
import tensorflow as tf

# Dynamic range quantization
input_model = sys.argv[1]
output_model = "name-the-quantized-model-here.tflite" if len(sys.argv) < 3 else f"{sys.argv[2]}.tflite"
#model = tf.keras.models.load_model("put-the-ori-trained-keras-model-here.hdf5")
model = tf.keras.models.load_model(input_model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()

# Save the converted model to TF lite file
# with open("name-the-quantized-model-here.tflite", "wb") as output_file:
with open(output_model, "wb") as output_file:
    output_file.write(tflite_quant_model)
