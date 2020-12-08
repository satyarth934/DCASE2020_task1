##### For testing the original keras model, which is saved as .hdf5 format.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import h5py
import librosa
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import soundfile as sound
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import sys
sys.path.append("..")

from utils import *
from funcs import *

import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

data_path = '../../task1b/data_2020/'
test_csv = data_path + 'evaluation_setup/fold1_test.csv'
feat_path = '../../data/features/logmel128_scaled/'
# experiments = f'exp_custom_expno_{wandb.config.exp_no}'
experiments = 'exp_mobnet'
# model_path = '../train/exp_custom_expno_2/model-04-0.7302.hdf5'
model_path = '../train/wandb/latest-run/files/model-best.h5'


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

	# data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
	data_val, y_val = load_data_2020(feat_path, test_csv, num_freq_bin, 'logmel')
	y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

	print("data_val:", data_val.shape)

	best_model = tf.keras.models.load_model(model_path)
	preds = best_model.predict(data_val)

	# General information about the model
	get_inference_metrics(model=best_model, data=data_val[0], label=y_val_onehot[0])

	y_pred_val = np.argmax(preds,axis=1)

	over_loss = log_loss(y_val_onehot, preds)
	overall_acc = np.sum(y_pred_val==y_val) / data_val.shape[0]

	print(y_val_onehot.shape, preds.shape)
	np.set_printoptions(precision=3)

	print("\n\nTest acc: ", "{0:.3f}".format(overall_acc))
	print("Test log loss:", "{0:.3f}".format(over_loss))

	conf_matrix = confusion_matrix(y_val,y_pred_val)
	print("\n\nConfusion matrix:")
	print(conf_matrix)
	conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
	recall_by_class = np.diagonal(conf_mat_norm_recall)
	mean_recall = np.mean(recall_by_class)

	dev_test_df = pd.read_csv(test_csv,sep='\t', encoding='ASCII')
	ClassNames = np.unique(dev_test_df['scene_label'])

	print("Class names:", ClassNames)
	print("Per-class test acc: ",recall_by_class, "\n\n")


if __name__ == '__main__':
	main()
