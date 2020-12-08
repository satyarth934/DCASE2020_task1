##### For testing the original keras model, which is saved as .hdf5 format.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("../eval_2")

import time
import h5py
import keras
import pickle
import librosa
import scipy.io
import tensorflow
import numpy as np
import pandas as pd
import soundfile as sound
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from keras.models import Model
import tensorflow as tf

import utils
import funcs

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


data_path = '../../task1b/data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
# feat_path = 'features/logmel128_scaled_d_dd/'
feat_path = '../../data/features/logmel128_scaled_singlechannel_sr8000_10secs_nodelta/'
wandb_run_dir = "run-20201206_222303-no6fl18k"
model_path = f'wandb/{wandb_run_dir}/files/model-best.h5'


def main():
	num_freq_bin = 128
	num_classes = 3

	data_val, y_val = utils.load_data_2020(feat_path, train_csv, num_freq_bin, 'logmel')
	y_val_onehot = keras.utils.to_categorical(y_val, num_classes)

	print(data_val.shape)
	print(y_val.shape)

	best_model = keras.models.load_model(model_path)

	best_model.summary()
	# print(best_model.layers[-14])
	# print(best_model.layers[-14].name)
	# print(best_model.layers[-14].input)
	# print(best_model.layers[-14].output)
	# print("==============================")
	# print(best_model.layers[0])
	# print(best_model.layers[0].name)
	# print(best_model.layers[0].input)
	# print(best_model.layers[0].output)

	featurizer = Model(best_model.layers[0].output, best_model.layers[-14].output)
	featurizer.summary()

	features = featurizer.predict(data_val)
	print("features:", features.shape)

	feature_label_dict = {"data": features, "labels": y_val}
	with open('mobnet_smaller3_features.pickle', 'wb') as handle:
		pickle.dump(feature_label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main()
