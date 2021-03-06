
# z=ax+by+c を求める。

x = [9.83, -9.97, -3.91, -3.94, -13.67, -14.04, 4.81, 7.65, 5.50, -3.34]
y = [-5.50, -13.53, -1.23, 6.07, 1.94, 2.79, -5.43, 15.57, 7.26, 1.34]
z = [635.99, 163.78, 86.94, 245.35, 1132.88, 1239.55, 214.01, 67.94, -1.48, 104.18]

import numpy as np
from scipy import linalg as LA

N = len(x)
G = np.array([x, y, np.ones(N)]).T
print(G)
print(G.T)
# num.dotを用いて内積を求める
result = LA.solve(G.T.dot(G), G.T.dot(z))

print("g")
print(G.T.dot(G))

print("z")
print(G.T.dot(ｚ))

print("result")
print(result)
