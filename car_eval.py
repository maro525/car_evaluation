# -*- coding: utf-8 -*-

# ライブラリ読み込み
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier


# データ読み込み
df = pd.read_csv('car.data.txt')
df = df.apply(preprocessing.LabelEncoder().fit_transform)

# Xに属性のデータをいれて、yに答えのラベルを入れる
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

# データを訓練用とテスト用に分ける
propotion = 0.8  # 訓練用のデータの割合
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = propotion)

clf = KNeighborsClassifier(n_neighbors=3) # KNeighborsClassifierの機械学習モデルをセット
clf.fit(X_train, y_train) # 機械学習

accuracy = clf.score(X_test, y_test) # テスト用データで答えを予測すし、その正解率を算出
print(accuracy)
