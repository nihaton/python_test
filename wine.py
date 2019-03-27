import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sciket-learnで重回帰分析
# https://pythondatascience.plavox.info/scikit-learn/%E7%B7%9A%E5%BD%A2%E5%9B%9E%E5%B8%B0
# y= b1x1 + b2x2 +b3x3 + ・・・ +bkxk + e(誤差)

# CSVファイル読み込み
wine = pd.read_csv("winequality-red.csv", sep=";")
wine.head

# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
clf = linear_model.LinearRegression()

# データフレームの各列を正規化
wine2 = wine.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
wine2.head()

# 説明変数に "quality (品質スコア以外すべて)" を利用
wine2_except_quality = wine2.drop("quality", axis=1)
X = wine2_except_quality.as_matrix()

# 目的変数に "quality (品質スコア)" を利用
Y = wine2['quality'].as_matrix()

# 予測モデルを作成（X:説明変数、Y:目的変数）
clf.fit(X, Y)

# 偏回帰係数
#print(pd.DataFrame({"Name":wine2_except_quality.columns,
#                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients') )

# 偏回帰係数
print(pd.DataFrame({"Name":wine2_except_quality.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
