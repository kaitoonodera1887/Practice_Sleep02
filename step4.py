import pandas as pd
import numpy as np
from scipy.signal import welch

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

# 周波数帯域
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30)
}

# サンプリング周波数
fs = 250

freq_features_list = []
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

    # Welch法でパワースペクトル計算
    freqs, psd = welch(eeg_epoch, fs=fs, nperseg=512)  # nperseg=fs -> 1秒ごとの窓
    
    epoch_features = []
    total_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 30)])  # 0.5-30Hz全体のパワー
    
    for band, (f_low, f_high) in freq_bands.items():
        band_power = np.sum(psd[(freqs >= f_low) & (freqs < f_high)])
        band_rel = band_power / total_power  # 相対パワー
        epoch_features.extend([band_power, band_rel])
    
    freq_features_list.append(epoch_features)

print(freq_features_list)

features_df = pd.DataFrame(features_list, columns=['eeg_mean','eeg_std','eeg_rms','emg_mean','emg_std','emg_rms'])

print(features_df.head(10))

# 周波数特徴量をDataFrameに変換
freq_columns = []
for band in freq_bands.keys():
    freq_columns.append(f'{band}_abs')  # 絶対パワー
    freq_columns.append(f'{band}_rel')  # 相対パワー

freq_features_df = pd.DataFrame(freq_features_list, columns=freq_columns)

# 元の特徴量と結合
features_df = pd.concat([features_df, freq_features_df], axis=1)
print(features_df.head())


# ラベルCSVを読み込む
label_file = "labels.csv"
labels_df = pd.read_csv(label_file)

# Epoch列のみを使用して結合
labels_df = labels_df[['Epoch', 'No.']]

print(labels_df.head())

# 特徴量DataFrameにラベル列を追加
# 注意: 特徴量の行数とラベルの行数を揃える
features_df['label'] = labels_df['No.'][:len(features_df)].values

print(features_df.head())

