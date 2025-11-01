import pandas as pd
import numpy as np

file_path = "sample_data.csv"
df = pd.read_csv(file_path)

print(df.shape)
print(df.info)
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())

eeg = df["EEG"].values
emg = df["EMG"].values

fs = 250  # サンプリング周波数
epoch_length_sec = 4
samples_per_epoch = fs * epoch_length_sec

# エポック数
num_epochs = len(eeg) // samples_per_epoch

print(f"num_epochs : {num_epochs}")

# 特徴量を格納するリスト
features_list = []


for i in range(num_epochs):
    start = i * samples_per_epoch
    end = start + samples_per_epoch

    if i == 0:
        print(f"start time : {start}")
        print(f"end time : {end}")
    
    eeg_epoch = eeg[start:end]
    emg_epoch = emg[start:end]
    
    # 基本統計量
    eeg_mean = np.mean(eeg_epoch)
    eeg_std = np.std(eeg_epoch)
    eeg_rms = np.sqrt(np.mean(eeg_epoch**2))
    
    emg_mean = np.mean(emg_epoch)
    emg_std = np.std(emg_epoch)
    emg_rms = np.sqrt(np.mean(emg_epoch**2))
    
    features_list.append([eeg_mean, eeg_std, eeg_rms, emg_mean, emg_std, emg_rms])

#print(features_list)

features_df = pd.DataFrame(features_list, columns=['eeg_mean','eeg_std','eeg_rms','emg_mean','emg_std','emg_rms'])

print(features_df.head(10))



