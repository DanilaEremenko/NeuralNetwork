import matplotlib.pyplot as plt
import random as rand
import math as m


def isAcceptableOval(x, y, xCenter=30, yCenter=30, radVert=10, radHor=10):
    if not m.pow((x - xCenter) / radHor, 2) + m.pow((y - yCenter) / radVert, 2) <= 1:
        return False
    return True


def isAcceptableRect(x, y, xMin=0, yMin=0, xMax=0, yMax=0):
    if not (range(xMin, xMax).__contains__(x) and range(yMin, yMax).__contains__(y)):
        return False
    return True


if __name__ == '__main__':
    pointNumber = 1500
    xCoor = []
    yCoor = []

    # FirstFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while (not isAcceptableRect(x, y, xMin=30, yMin=30, xMax=70, yMax=70)):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    plt.plot(xCoor, yCoor, '.r')

    xCoor.clear()
    yCoor.clear()


    # SecondFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while (not isAcceptableOval(x, y, xCenter=100, yCenter=120, radHor=30, radVert=30) or
               isAcceptableOval(x, y, xCenter=130, yCenter=120, radHor=30, radVert=30)
               or isAcceptableRect(x, y, xMin=60, yMin=110, xMax=80, yMax=130)):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    plt.plot(xCoor, yCoor, '.b')

    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
