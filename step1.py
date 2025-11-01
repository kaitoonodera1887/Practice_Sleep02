import pandas as pd  # データ操作用ライブラリ

# CSVファイルを読み込む
file_path = "sample_data.csv"  # 実際のファイルパスに置き換える
data = pd.read_csv(file_path)

print("data.shape : ", data.shape)
print("data.info : ", data.info)
print("data.isnull : ", data.isnull().sum())

# データの最初の5行を確認
print(data.head())

# 列名の確認
print(data.columns)

# データ型の確認
print(data.dtypes)


import numpy as np  # 数値計算用ライブラリ

# EEG列の統計量
eeg = data['EEG'].values  # DataFrameからnumpy配列に変換
print("eeg dtype : ", type(eeg))
eeg_mean = np.mean(eeg)
print("eeg_mean dtype : ", type(eeg_mean))
eeg_std = np.std(eeg)
eeg_rms = np.sqrt(np.mean(eeg**2))

# EMG列の統計量
emg = data['EMG'].values
emg_mean = np.mean(emg)
emg_std = np.std(emg)
emg_rms = np.sqrt(np.mean(emg**2))

# 結果表示
print("EEG mean:", eeg_mean, "EEG std:", eeg_std, "EEG RMS:", eeg_rms)
print("EMG mean:", emg_mean, "EMG std:", emg_std, "EMG RMS:", emg_rms)