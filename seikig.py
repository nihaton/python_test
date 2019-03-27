import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4,4, 100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

plt.ylim(0,0.5)
plt.plot(x,y)
plt.show()

#https://qiita.com/kenmatsu4/items/351284ef430bcfd2c8ed
