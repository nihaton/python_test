# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#http://kenbo.hatenablog.com/entry/2018/11/30/091511

def main():

#入力データべた書き
 X = np.array([1,2,3,4,5])
 print(X.dtype)
 Y = [1.1, 2.1, 2.8, 4.3, 5.1]

 #ones:要素が1の配列を生成する
 A = np.array([X,np.ones(len(X))])
 #行列を縦に変換
 A = A.T
 print(A)
 #線形回帰(np.linalg.lstsq)を実行してa:傾き、b:切片を取得。
 a,b = np.linalg.lstsq(A,Y)[0]
 #print("a(傾き):{:.5f}".format(a))
 print("a(傾き):{:}".format(a))
 print("b(切片)：{:}".format(b))
 #X,Y生データのプロット
 plt.plot(X,Y,"ro")
 #y=a*X;bの回帰式をプロット
 maxy = max(Y)
 plt.text(1, maxy-1, "y={:.2f}x+{:.2f}".format(a,b), fontsize=14)
 plt.plot(X,(a*X+b),"b--")
 plt.grid()
 plt.show()

if __name__ == '__main__':
     main()
