# coding: UTF-8
import numpy as np

def stat(obj, exp):

    n = exp.shape[1]

    exp = np.vstack([np.ones(n), exp]) # 定数項、説明変数

    coef = np.linalg.lstsq(exp.T, obj)[0] # 偏回帰係数

    return coef

if __name__ == '__main__':

    f = (45, 38, 41, 34, 59, 47, 35, 43, 54, 52)                     # 配向度
    t = (17.5, 17.0, 18.5, 16.0, 19.0, 19.5, 16.0, 18.0, 19.0, 19.5) # 温度
    p = (30, 25, 20, 30, 45, 35, 25, 35, 35, 40)                     # 圧力

    obj = np.array(f)      # 目的変数
    exp = np.array([t, p]) # 説明変数

    b0, bt, bp = stat(obj, exp)
    print ("---")
    print ("重回帰式: 配向度 = {0:f} + {1:f}*温度 + {2:f}*圧力".format(b0, bt, bp))
