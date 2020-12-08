import os
import sys
import glob
import tensorflow as tf


def main():
	models_path = "../train_2/wandb" if len(sys.argv) < 2 else sys.argv[1]
	inp_model_paths = glob.glob(f"{models_path}/*/files/*.h5")
	# print(len(inp_model_paths))
	# print(inp_model_paths[0])
	# print(inp_model_paths[-1])

	for inp_model_path in inp_model_paths:
		output_model_path = os.path.join(os.path.dirname(inp_model_path), os.path.basename(inp_model_path).replace(".h5", "-quantized.tflite"))
		if os.path.exists(output_model_path):
			continue

		model = tf.keras.models.load_model(inp_model_path)
		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		# converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		tflite_quant_model = converter.convert()

		# Save the converted model to TF lite file
		# with open("name-the-quantized-model-here.tflite", "wb") as output_file:
		with open(output_model_path, "wb") as output_file:
		    output_file.write(tflite_quant_model)


if __name__ == '__main__':
	main()