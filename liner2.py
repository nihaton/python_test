# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#http://palloc.hateblo.jp/entry/2016/01/11/170424

data = [(2, 3), (4, 7), (9, 11)]
a = np.matrix(data)
b = a[:, 1].copy()
a[:, 1] = 1

print(a)
print(b)
print("**************")
#縦ベクトルを横ベクトルに変換
print(a.T )
print("**************")
# 2*2 + 4*4 + 9*9 = 101
# 2*1 + 4*1 + 9*1 = 15
# 1*2 + 1*4 + 1*9 = 15
# 1*1 + 1*1 + 1*1 = 3
#２次元ベクトルに変換
print(a.T * a )
print("**************")
#アスタリスク２つはべき乗
#-1は逆行列
print((a.T * a)**-1)
print("**************")
# 係数ベクトルを求める
x = ((a.T * a)**-1) * (a.T * b)
x= np.array(x)
print(x)

# matplotlibで描画
datax, datay = np.split(np.array(data), 2, axis=1)
maxx, maxy = max(datax), max(datay)
plt.xlim(0, maxx*1.1)
plt.ylim(0, maxy*1.1)
plt.scatter(datax, datay)
plt.plot([0, maxx*1.1], [x[1], maxx*x[0]*1.1 + x[1]], color="r")
plt.text(1, maxy, "y={:.2f}x+{:.2f}".format(*x[:, 0]), fontsize=14)
plt.show()
