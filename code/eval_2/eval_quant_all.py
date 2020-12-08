##### For testing the original keras model, which is saved as .hdf5/.h5 format.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.dont_write_bytecode = True
sys.path.append("..")
if len(sys.argv) < 2:
	print("ERROR: Enter yaml file for runs and dataset associations.")
	sys.exit(0)

import tqdm
import glob
import time
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import utils
import funcs

np.set_printoptions(precision=3)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# run_data_dict = {"20zf33ta": "logmel128_scaled_singlechannel_sr8000_10secs_nodelta",
# 					"no6fl18k": "logmel128_scaled_singlechannel_sr8000_10secs_nodelta",
# 					"9kebg1uj": "logmel128_scaled_singlechannel_sr8000_10secs",
# 					"lpjydpz2": "logmel128_scaled_singlechannel_sr12000_5secs",
# 					"358h4nim": "logmel128_scaled_singlechannel_sr24000_5secs",
# 					"2wvyewts": "logmel128_scaled_singlechannel_sr24000_5secs",
# 					"1nmj0zhj": "logmel128_scaled_singlechannel_sr24000_10secs",
# 					"165xrxp2": "logmel128_scaled_singlechannel_sr32000_5secs",
# 					"7hxzif83": "logmel128_scaled",
# 					"2a5pwrd1": "logmel128_scaled",
# 					"1abfsyiw": "logmel128_scaled",
# 					"6gp9ibcm": "logmel128_scaled",
# 					}

run_data_dict = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)


# Evaluate the models
## Define a helper function to evaluate the TFlite model using test dataset. 
def evaluate_model(interpreter, test_images, test_labels, num_class, is_eval=False):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Run predictions on every image in the test dataset.
    prediction_digits = []
    pred_output_all = np.empty([1,num_class])
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_index)
        pred_output = output[0]
        pred_output.reshape([1,num_class])
        pred_output_all = np.vstack((pred_output_all, pred_output))
        digit = np.argmax(output[0])
        prediction_digits.append(digit)
        
        
    pred_output_all = pred_output_all[1:,:]
    
    if is_eval:
        return None, pred_output_all, prediction_digits
    else:
        # Compare prediction results with ground truth labels from the validation set to calculate accuracy.
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)
    
        return accuracy, pred_output_all, prediction_digits


def main():
	outf = open(sys.argv[2], "w")

	data_path = '../../task1b/data_2020/'
	test_csv = data_path + 'evaluation_setup/fold1_test.csv'
	feat_path_parent = '../../data/features'

	# Evaluate for each run
	for run in tqdm.tqdm(run_data_dict):
		# find data using data
		data_dir = run_data_dict[run]
		feat_path = f"{feat_path_parent}/{data_dir}"
		num_freq_bin = 128
		num_classes = 3
		data_val, y_val = utils.load_data_2020(feat_path, test_csv, num_freq_bin, 'logmel')
		y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

		print("data_val:", data_val.shape)
		print("y_val:", y_val.shape)

		# find model using run
		wandb_dir = "../train_2/wandb"
		model_file = f"{wandb_dir}/{[run_dir for run_dir in os.listdir(wandb_dir) if run in run_dir][0]}/files/model-best-quantized.tflite"
		print("===>>>>>>>>>>", model_file)

		outf.write(f"===== MODEL: {model_file} =====")
		outf.write("\n")
		outf.write(f"----- DATA: {feat_path} -----")
		outf.write("\n")

		# Load the model into an interpreter and predict
		interpreter_quant = tf.lite.Interpreter(model_path=model_file)
		interpreter_quant.allocate_tensors()
		overall_acc, preds, preds_class_idx = evaluate_model(interpreter=interpreter_quant, 
																test_images=data_val, 
																test_labels=y_val, 
																num_class=num_classes)
		
		over_loss = log_loss(y_val_onehot, preds)

		# Map the class index (0,1,2) to class name (indoor, outdoor, transportation).
		preds_class_idx = np.array(preds_class_idx)
		preds_class_idx = np.reshape(preds_class_idx, (len(y_val),1))
		preds_class = np.where(preds_class_idx > 1, 'transportation', 
		                       (np.where(preds_class_idx < 1, 'indoor', 'outdoor')))

		# Output the prediction resutls to a csv file with a format required by the DCASE.
		test_output_df = pd.DataFrame({
		                               'scene_label': preds_class[:,0],
		                               'indoor': preds[:,0],
		                               'outdoor': preds[:,1],
		                               'transportation': preds[:,2]})
		                               
		print(f"y_val_onehot:{y_val_onehot.shape}\tpreds:{preds.shape}")
		
		outf.write("\n")
		outf.write(f"Test acc: {overall_acc:.3f}")
		outf.write("\n")
		outf.write(f"Test log loss: {over_loss:.3f}")
		outf.write("\n")

		y_pred_val = np.argmax(preds, axis=1)
		conf_matrix = confusion_matrix(y_val, y_pred_val)
		outf.write("Confusion matrix:")
		outf.write("\n")
		outf.write(str(conf_matrix))
		outf.write("\n")

		conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
		recall_by_class = np.diagonal(conf_mat_norm_recall)
		mean_recall = np.mean(recall_by_class)

		dev_test_df = pd.read_csv(test_csv, sep='\t', encoding='ASCII')
		ClassNames = np.unique(dev_test_df['scene_label'])

		outf.write(f"Class names: {ClassNames}")
		outf.write("\n")
		outf.write(f"Per-class test acc: {recall_by_class}")
		outf.write("\n")

		# Inference Time
		start = time.time()
		acc, pred, pred_class_idx = evaluate_model(interpreter=interpreter_quant, test_images=np.expand_dims(data_val[0], axis=0), test_labels=np.expand_dims(y_val[0], axis=0), num_class=num_classes, is_eval=True)
		# pred = model.predict(np.expand_dims(data_val[0], axis=0))
		inference_time = time.time() - start
		outf.write(f"inference_time: {inference_time}")
		outf.write("\n\n")

		time.sleep(5)


if __name__ == '__main__':
	main()
