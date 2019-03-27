import pandas as pd
import numpy as np

#タイタニック号
#https://www.codexa.net/kaggle-titanic-beginner/

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

test_shape = test.shape
train_shape = train.shape

print(test_shape)
print(train_shape)

test.describe()
train.describe()

def kesson_table(df):
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns

kesson_table(train)
kesson_table(test)

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

kesson_table(train)

#男性を1、女性を0に設定
train["Sex"] = train["Sex"].apply(lambda sex:1 if sex=="male" else 0)
train["Embarked"] = train["Embarked"].apply(lambda embarked:0 if embarked=="S" else 1 if embarked=="C" else 2)
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())

train.head(10)
#男性を1、女性を0に設定
test["Sex"] = test["Sex"].apply(lambda sex:1 if sex=="male" else 0)
test["Embarked"] = test["Embarked"].apply(lambda embarked:0 if embarked=="S"  else 1 if embarked=="C" else 2)
test.Fare[152] = test.Fare.median()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

test.head(10)

 # scikit-learnのインポートをします
from sklearn import tree

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

# 予測データのサイズを確認
my_prediction.shape

 # 予測データのサイズを確認
my_prediction.shape

print(my_prediction)

PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
