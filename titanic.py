import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#https://ja.stackoverflow.com/questions/42701/python%e3%82%92%e4%bd%bf%e3%81%a3%e3%81%a6%e3%83%ad%e3%82%b8%e3%82%b9%e3%83%86%e3%82%a3%e3%83%83%e3%82%af%e5%9b%9e%e5%b8%b0%e3%81%97%e3%81%9f%e3%81%a8%e3%81%8d%e3%81%aep%e5%80%a4

def load_file_train():
    train_df = pd.read_csv("input/train.csv")
    cols = ["Pclass", "Sex", "Age"]
    #男性を1、女性を0に設定
    train_df["Sex"] = train_df["Sex"].apply(lambda sex:1 if sex=="male" else 0)
    #年齢がないデータの年齢を平均年齢にする
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
    train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].mean())
    survived = train_df["Survived"].values
    data = train_df[cols].values
    return survived, data

def load_file_test():
    train_df = pd.read_csv("input/test.csv")
    cols = ["Pclass", "Sex", "Age"]
    #男性を1、女性を0に設定
    train_df["Sex"] = train_df["Sex"].apply(lambda sex:1 if sex=="male" else 0)
    #年齢がないデータの年齢を平均年齢にする
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
    train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].mean())
    data = train_df[cols].values
    return data

survived,data_train = load_file_train()
model = LogisticRegression()
model.fit(data_train, survived)

data_test = load_file_test()

predicted = model.predict(data_test)

PassengerId = np.array(predicted["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(predicted, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
