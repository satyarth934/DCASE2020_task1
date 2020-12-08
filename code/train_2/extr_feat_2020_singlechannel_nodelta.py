import os
import sys
import glob
import time
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


def cal_deltas(X_in):
    X_out = (X_in[:, 2:, :] - X_in[:, :-2, :]) / 10.0
    X_out = X_out[:, 1:-1, :] + (X_in[:, 4:, :] - X_in[:,:-4,:]) / 5.0
    return X_out


def main():

    wavpath = glob.glob("../../data/*/*/audio/*.wav")
    wavpath = [("/".join(wp.split("/")[:-2]) + "/", "/".join(wp.split("/")[-2:])) for wp in wavpath]

    output_path = '../../data/features/logmel128_scaled_singlechannel_sr2000_freqbins64_nfft512_10secs_nodelta'
    feature_type = 'logmel'

    # sr = 48000
    sr = 2000
    duration = 10
    # duration = 5
    # num_freq_bin = 128
    num_freq_bin = 64
    # num_fft = 2048
    num_fft = 512
    hop_length = int(num_fft / 2)
    num_time_bin = int(np.ceil(duration * sr / hop_length))
    # num_channel = 2
    num_channel = 1

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
    # wavpath = data_df['filename'].tolist()

    # FOR TIME PROFILING
    mel_freq_time = 0

    # wavpath = wavpath[:50]
    
    for i in range(len(wavpath)):
    # for i in range(50):
        # stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
        file_path = wavpath[i][0]
        wp = wavpath[i][1]

        start_time = time.time()
        stereo, fs = sound.read(file_path + wp, stop=duration*sr)
        logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        for ch in range(num_channel):
            logmel_data[:,:,ch]= librosa.feature.melspectrogram(stereo[:,ch], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

        logmel_data = np.log(logmel_data+1e-8)
        
        feat_data = logmel_data
        feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
        mel_freq_time += (time.time() - start_time)

        feature_data = {'feat_data': feat_data,}

        cur_file_name = output_path + wp[5:-3] + feature_type
        pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print("feat_data:", feat_data.shape)
    print(f"Mel frequency time for sr{sr}, channels{num_channel}, {duration} seconds:\n{(mel_freq_time/len(wavpath))}")
    print()

  
if __name__ == '__main__':
    main()