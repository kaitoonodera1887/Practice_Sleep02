import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

features_path = "features.csv"
features_df = pd.read_csv(features_path)

print(features_df.shape)
print(features_df.dtypes)
print(features_df.head())

labels_path = "labels.csv"
labels_df = pd.read_csv(labels_path, dtype=str, sep=None, engine='python')  # flexible reader

print(labels_df.shape)
print(labels_df.dtypes)
print(labels_df.head())

# 2) normalize label columns detection (simple)
cols = [c.strip().lower() for c in labels_df.columns] #delete literature
print('label file columns:', labels_df.columns.tolist())

# assume labels file has 'epoch' and a label column (e.g., 'no.' or 'classification')
# try common names:
epoch_col = None
for name in ('epoch','epoch_num','no','no.','epoch number'):
    if name in cols:
        epoch_col = labels_df.columns[cols.index(name)]
        break

label_col = None
for name in ('no.','no','label','classification','class','stage'):
    if name in cols:
        label_col = labels_df.columns[cols.index(name)]
        break

print('detected epoch_col =', epoch_col, ' label_col =', label_col)

# 3) build label df with integer epoch
lab = labels_df[[epoch_col, label_col]].copy()
lab.columns = ['epoch_raw', 'label_raw']
lab['epoch_num'] = pd.to_numeric(lab['epoch_raw'].astype(str).str.strip(), errors='coerce').astype('Int64')
lab = lab[lab['epoch_num'].notna()].copy()
lab['epoch_num'] = lab['epoch_num'].astype(int)
print(lab)
print(lab.dtypes)


# map W/N/R -> WAKE/NREM/REM
def map_label(s):
    s = str(s).strip().upper()
    if s == 'W': return 'WAKE'
    if s == 'N': return 'NREM'
    if s == 'R': return 'REM'
    return s

lab['label'] = lab['label_raw'].apply(map_label)

# 4) merge by epoch_num (features must have epoch_num 1-based)
if 'epoch_num' not in features_df.columns:
    features_df = features_df.reset_index(drop=True)
    features_df['epoch_num'] = np.arange(1, len(features_df)+1)

merged = pd.merge(features_df, lab[['epoch_num','label']], on='epoch_num', how='left')
print(merged.head())
print('merged shape:', merged.shape)

# 5) build X and y (rows that have labels)
labeled = merged[~merged['label'].isna()].copy()
print(labeled.head())
print('number of labeled epochs:', len(labeled))
X = labeled.drop(columns=['epoch_num','start_time','end_time','label'], errors='ignore')
y = labeled['label'].values

# 6) encode labels and split
le = LabelEncoder()
y_enc = le.fit_transform(y)
print('classes:', le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

print('X_train shape:', X_train.shape, 'X_test shape:', X_test.shape)
print('y_train distribution:', np.bincount(y_train))
print('y_test distribution:', np.bincount(y_test))

# save for the next step if you like
import joblib
joblib.dump((X_train, X_test, y_train, y_test, le), 'train_split.pkl')

print(features_df.columns.tolist())