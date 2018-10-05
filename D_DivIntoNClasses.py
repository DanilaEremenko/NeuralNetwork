import matplotlib.pyplot as plt
import random as rand
import math as m


def isOval(x, y, xCenter=30, yCenter=30, radVert=10, radHor=10):
    if not m.pow((x - xCenter) / radHor, 2) + m.pow((y - yCenter) / radVert, 2) <= 1:
        return False
    return True


def isRect(x, y, xMin=0, yMin=0, xMax=0, yMax=0):
    if not (range(xMin, xMax).__contains__(x) and range(yMin, yMax).__contains__(y)):
        return False
    return True


def isTriangle(x, y, x1=0, y1=0, x2=0, y2=0, x3=0, y3=0):
    sign1 = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y)
    sign2 = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y)
    sign3 = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y)

    # Нормализация
    sign1 /= m.fabs(sign1)
    sign2 /= m.fabs(sign3)
    sign3 /= m.fabs(sign2)

    if (int(sign1) & int(sign2) & int(sign3)) == 0:
        return False

    return True


if __name__ == '__main__':
    pointNumber = 1000
    xCoor = []
    yCoor = []

    # FirstFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while (not isRect(x, y, xMin=30, yMin=30, xMax=70, yMax=70)):
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
        while (not isOval(x, y, xCenter=100, yCenter=120, radHor=30, radVert=30) or
               isOval(x, y, xCenter=130, yCenter=120, radHor=30, radVert=30)
               or isRect(x, y, xMin=60, yMin=110, xMax=80, yMax=130)):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    plt.plot(xCoor, yCoor, '.b')

    xCoor.clear()
    yCoor.clear()

    # # ThirdFigure
    # for i in range(0, pointNumber):
    #     x = rand.randint(0, 200)
    #     y = rand.randint(0, 200)
    #     while (not isTriangle(x, y, x1=10, y1=10, x2=20, y2=20, x3=10, y3=30)):
    #         x = rand.randint(0, 200)
    #         y = rand.randint(0, 200)
    #     xCoor.append(x)
    #     yCoor.append(y)
    #
    # plt.plot(xCoor, yCoor, '.g')

    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
