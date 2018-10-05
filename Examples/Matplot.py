import matplotlib.pyplot as plt
import numpy as np

xList = np.random.randint(10, 90, size=100)
yList = np.random.randint(10, 90, size=100)

xList.sort()
yList.sort()

plt.plot(xList, yList, '--r')

xList = np.random.randint(10, 90, size=100)
yList = np.random.randint(10, 90, size=100)

xList.sort()
yList.sort()

plt.plot(xList, yList, '--b')


plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title("arrays")
plt.show()
