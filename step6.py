import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# 1) 学習データの読み込み
X_train, X_test, y_train, y_test, le = joblib.load('train_split.pkl')
print("データ読み込み完了")

# 2) モデルの作成
# RandomForestClassifierは多数の決定木の「多数決」で分類する
clf = RandomForestClassifier(
    n_estimators=200,        # 木の数（多いほど精度↑ただし学習時間↑）
    max_depth=None,          # 木の深さ（Noneは自動）
    class_weight='balanced', # クラス不均衡（WAKEが多い）の補正
    random_state=42,
    n_jobs=-1                # CPU全コア使用
)

# 3) モデルの学習
clf.fit(X_train, y_train)
print("学習完了")

# 4) 評価
y_pred = clf.predict(X_test)
print("分類レポート:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("Macro F1 score:", f1_score(y_test, y_pred, average='macro'))


# 5) 特徴量の重要度の確認（モデル内蔵のimportance）
feature_names = [f"f{i}" for i in range(X_train.shape[1])]
importances = clf.feature_importances_
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False)
print("\n上位特徴量:")
print(imp_df.head(10))

# 6) 可視化（棒グラフ）
plt.figure(figsize=(8,5))
plt.barh(imp_df['Feature'][:10][::-1], imp_df['Importance'][:10][::-1])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (RandomForest)")
plt.savefig('Importance.png')
plt.show()


# 7) （任意）Permutation importance — 特徴をランダムに入れ替えて性能低下を測る
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()[::-1]

print("\nPermutation Importance 上位5:")
for idx in perm_sorted_idx[:5]:
    print(f"{feature_names[idx]}: {result.importances_mean[idx]:.4f}")

# 保存（あとでGUIや別スクリプトで使うため）
joblib.dump(clf, 'sleep_rf_model.pkl')
print("モデル保存完了: sleep_rf_model.pkl")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

