##### For testing the original keras model, which is saved as .hdf5 format.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import h5py
import keras
import librosa
import scipy.io
import tensorflow
import numpy as np
import pandas as pd
import soundfile as sound
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import sys
sys.path.append("..")
from utils import *
from funcs import *

import tensorflow as tf
# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


data_path = '../../task1b/data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
# feat_path = 'features/logmel128_scaled_d_dd/'
feat_path = '../../data/features/logmel128_scaled/'
wandb_run_dir = "run-20201205_164213-2jjs9gun"
model_path = f'../train_2/wandb/{wandb_run_dir}/files/model-best.h5'


def get_inference_metrics(model, data, label):
	X = frequency_masking(data)
	X = time_masking(X)

	start = time.time()
	pred = model.predict(np.expand_dims(X, axis=0))

	print(f"\nInference time: {time.time() - start} seconds")
	print(f"Size of the model {sys.getsizeof(model)}")
	print(f"Model prediction: {pred}\t Ground Truth: {label}\n")


def main():
	num_freq_bin = 128
	num_classes = 3

	data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
	y_val_onehot = keras.utils.to_categorical(y_val, num_classes)

	print(data_val.shape)
	print(y_val.shape)

	best_model = keras.models.load_model(model_path)
	preds = best_model.predict(data_val)

	# General information about the model
	get_inference_metrics(model=best_model, data=data_val[0], label=y_val[0])

	y_pred_val = np.argmax(preds,axis=1)

	over_loss = log_loss(y_val_onehot, preds)
	overall_acc = np.sum(y_pred_val==y_val) / data_val.shape[0]

	print(y_val_onehot.shape, preds.shape)
	np.set_printoptions(precision=3)

	print("\n\nVal acc: ", "{0:.3f}".format(overall_acc))
	print("Val log loss:", "{0:.3f}".format(over_loss))

	conf_matrix = confusion_matrix(y_val,y_pred_val)
	print("\n\nConfusion matrix:")
	print(conf_matrix)
	conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
	recall_by_class = np.diagonal(conf_mat_norm_recall)
	mean_recall = np.mean(recall_by_class)

	dev_test_df = pd.read_csv(val_csv,sep='\t', encoding='ASCII')
	ClassNames = np.unique(dev_test_df['scene_label'])

	print("Class names:", ClassNames)
	print("Per-class val acc: ",recall_by_class, "\n\n")


if __name__ == '__main__':
	main()
