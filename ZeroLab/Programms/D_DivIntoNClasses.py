import matplotlib.pyplot as plt
import random as rand
import math as m


def isElipse(x, y, xCenter=30, yCenter=30, radVert=10, radHor=10):
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
    try:
        sign1 /= m.fabs(sign1)
        sign2 /= m.fabs(sign2)
        sign3 /= m.fabs(sign3)
    except ZeroDivisionError:
        return False

    if int(sign1) == int(sign2) == int(sign3):
        return True

    return False


if __name__ == '__main__':
    pointNumber = 300
    xCoor = []
    yCoor = []

    # FirstFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while not isRect(x, y, xMin=30, yMin=135, xMax=70, yMax=165) or \
                isTriangle(x, y, x1=30, y1=140, x2=60, y2=155, x3=65, y3=140):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    # normalize
    xCoor = list(map(lambda x: x / 200, xCoor))
    yCoor = list(map(lambda y: y / 200, yCoor))
    plt.plot(xCoor, yCoor, '.r')

    xCoor.clear()
    yCoor.clear()

    # SecondFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while (not isElipse(x, y, xCenter=150, yCenter=150, radHor=30, radVert=30) or
               isElipse(x, y, xCenter=180, yCenter=150, radHor=30, radVert=30)
               or isRect(x, y, xMin=110, yMin=140, xMax=130, yMax=160)):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    # normalize
    xCoor = list(map(lambda x: x / 200, xCoor))
    yCoor = list(map(lambda y: y / 200, yCoor))
    plt.plot(xCoor, yCoor, '.b')

    xCoor.clear()
    yCoor.clear()

    # ThirdFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while not isTriangle(x, y, x1=30, y1=30, x2=60, y2=60, x3=100, y3=30) \
                or isElipse(x, y, xCenter=30, yCenter=45, radVert=10, radHor=30) \
                or isElipse(x, y, xCenter=70, yCenter=30, radVert=15, radHor=5):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    # normalize
    xCoor = list(map(lambda x: x / 200, xCoor))
    yCoor = list(map(lambda y: y / 200, yCoor))
    plt.plot(xCoor, yCoor, '.g')

    xCoor.clear()
    yCoor.clear()

    # FourthFigure
    for i in range(0, pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 200)
        while not isRect(x, y, xMin=125, yMin=30, xMax=170, yMax=60) \
                or isElipse(x, y, xCenter=147, yCenter=20, radVert=20, radHor=10) \
                or isElipse(x, y, xCenter=147, yCenter=70, radVert=20, radHor=10) \
                or isElipse(x, y, xCenter=115, yCenter=45, radVert=10, radHor=20) \
                or isElipse(x, y, xCenter=180, yCenter=45, radVert=10, radHor=20):
            x = rand.randint(0, 200)
            y = rand.randint(0, 200)
        xCoor.append(x)
        yCoor.append(y)

    # normalize
    xCoor = list(map(lambda x: x / 200, xCoor))
    yCoor = list(map(lambda y: y / 200, yCoor))
    plt.plot(xCoor, yCoor, '.m', )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    SHOW = input('SHOW? [Y/n]')
    SAVE = input('SAVE? [Y/n]')

    if SAVE == 'Y':
        plt.savefig("../Pictures/4_DivIntoNClasses.png", dpi=200)
    if SHOW == 'Y':
        plt.show()
