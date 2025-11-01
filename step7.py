import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 1) 学習データの読み込み
X_train, X_test, y_train, y_test, le = joblib.load('train_split.pkl')
print("データ読み込み完了")

clf = RandomForestClassifier(random_state=42)

param_grid_1 = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [5, 10, 20, None]
}

param_grid_2 = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None]
}

grid_search = GridSearchCV(
    estimator = clf,
    param_grid = param_grid_2,
    cv = 3,
    scoring = 'f1_macro',
    n_jobs = -1
)

grid_search.fit(X_train, y_train)

print('Best Parametors', grid_search.best_params_)
print('Best Macro F1', grid_search.best_score_)