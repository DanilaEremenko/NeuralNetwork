import matplotlib.pyplot as plot

xList = []
yList = []

i = 0
while i < 100:
    xList.append(i)
    yList.append(i)
    i += 1


plot.bar(xList, yList)
plot.xlim(0, 200)
plot.ylim(0, 100)
plot.title("arrays")
plot.show()



