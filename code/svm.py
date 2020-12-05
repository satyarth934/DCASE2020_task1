import os
import sys
import glob
import tqdm
import shutil
import pathlib
import librosa
import soundfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepData():
	# ### READING DATA ### 
	some_num = 1  # between 1 and 21
	input_dir = f"../small_data/TAU-urban-acoustic-scenes-2020-3class-development.audio.{some_num}/TAU-urban-acoustic-scenes-2020-3class-development/audio/"
	files = glob.glob(f"{input_dir}/*")

	duration = 10
	sr = 48000
	stereo, fs = soundfile.read(files[0], stop=duration * sr)
	# print(stereo.shape)

	# Raw data
	fig, ax = plt.subplots(nrows=2)
	ax[0].plot(stereo[:,0], label="mic0")
	ax[1].plot(stereo[:,1], label="mic1")



	# ### MEL-TRANSFORM DATA ### 
	num_freq_bin = 128
	num_fft = 2048
	hop_length = int(num_fft / 2)
	num_time_bin = int(np.ceil(duration * sr / hop_length))
	num_channel = 2
	logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
	for ch in range(num_channel):
		logmel_data[:,:,ch]= librosa.feature.melspectrogram(stereo[:,ch], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

	logmel_data = np.log(logmel_data + 1e-8)
	fig, ax = plt.subplots(nrows=2)
	fig.suptitle("logmel_data")
	ax[0].imshow(logmel_data[:, :, 0], label="mic0")
	ax[1].imshow(logmel_data[:, :, 1], label="mic1")
	print("logmel_data:", logmel_data.shape)
	
	feat_data = logmel_data
	feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
	fig, ax = plt.subplots(nrows=2)
	fig.suptitle("logmel_data normalized")
	ax[0].imshow(feat_data[:, :, 0], label="mic0")
	ax[1].imshow(feat_data[:, :, 1], label="mic1")
	
	deltas = calDeltas(feat_data)
	fig, ax = plt.subplots(nrows=2)
	fig.suptitle("logmel_data one deltas")
	ax[0].imshow(deltas[:, :, 0], label="mic0")
	ax[1].imshow(deltas[:, :, 1], label="mic1")

	deltas_deltas = calDeltas(deltas)
	fig, ax = plt.subplots(nrows=2)
	fig.suptitle("logmel_data double deltas")
	ax[0].imshow(deltas_deltas[:, :, 0], label="mic0")
	ax[1].imshow(deltas_deltas[:, :, 1], label="mic1")

	feat_data = np.concatenate((feat_data[:, 4:-4, :], deltas[:, 2:-2, :], deltas_deltas), axis=2)

	print("feat_data:", feat_data.shape)
	
	plt.show()


def calDeltas(X_in):
	X_out = (X_in[:, 2:, :] - X_in[:, :-2, :]) / 10.0
	X_out = X_out[:, 1:-1, :] + (X_in[:, 4:, :] - X_in[:,:-4,:]) / 5.0
	return X_out


def main():
	csv_file = "/home/satyarth934/Projects/Fall2020/cmsc727/Project/DCASE2020_task1/task1b/train/exp_mobnet/fold1_train_balanceclass.csv"

	df = pd.read_csv(csv_file, sep="\t")
	print(df['scene_label'].value_counts())


if __name__ == '__main__':
	main()
