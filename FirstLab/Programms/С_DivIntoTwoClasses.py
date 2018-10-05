import matplotlib.pyplot as plt
import random as rand


def isAcceptableCoordinatesE(x, y, xBottom=10, xTop=60, yBottom=10, yTop=80):
    high = yTop - yBottom
    wide = int((yTop - yBottom) / 8)
    # Квадрат
    if (not range(xBottom, xTop).__contains__(x) and range(yBottom, yTop).__contains__(y)):
        return False

    # Вертикаль
    if (range(xBottom, xBottom + wide).__contains__(x) and range(yBottom, yBottom + high).__contains__(y)):
        return True
    # Низ
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yBottom, yBottom + wide).__contains__(y)):
        return True
    # Верх
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yTop - wide, yTop).__contains__(y)):
        return True
    # Середина
    elif (range(xBottom, xBottom + wide * 3).__contains__(x) and range(yBottom + 3 * wide,
                                                                       yBottom + 4 * wide).__contains__(y)):
        return True

    return False


def isAcceptableCoordinatesP(x, y, xBottom=100, xTop=150, yBottom=80, yTop=80):
    high = yTop - yBottom
    wide = int((yTop - yBottom) / 8)

    # Больший квадрат
    if (not range(xBottom, xTop).__contains__(x) and range(yBottom, yTop).__contains__(y)):
        return False

    # Вертикаль
    if (range(xBottom, xBottom + wide).__contains__(x) and
            range(yBottom, yBottom + high * 2).__contains__(y)):
        return True

    # Низ
    elif (range(xBottom, xBottom + high).__contains__(x) and
          range(yBottom + wide * 3, yBottom + wide * 4).__contains__(y)):
        return True

    # Верх
    elif (range(xBottom, xBottom + high).__contains__(x) and
          range(yTop - wide, yTop).__contains__(y)):
        return True

    # Правый край
    elif (range(xTop - wide, xTop).__contains__(x) and
          range(yBottom + 3 * wide, yTop).__contains__(y)):
        return True


if __name__ == '__main__':
    pointNumber = 2000
    xCoor = []
    yCoor = []

    for i in range(pointNumber):
        x = rand.randint(0, 200)
        y = rand.randint(0, 100)

        while (not isAcceptableCoordinatesE(x, y, xBottom=10, xTop=60, yBottom=10, yTop=100) and
               not isAcceptableCoordinatesP(x, y, xBottom=100, xTop=150, yBottom=10, yTop=100)):
            x = rand.randint(0, 200)
            y = rand.randint(0, 100)
        xCoor.append(x)
        yCoor.append(y)

    plt.xlim(0, 200)
    plt.ylim(0, 200)

    plt.plot(xCoor, yCoor, '.')



    SHOW = input('SHOW? [Y/n]')
    SAVE = input('SAVE? [Y/n]')


    if SAVE == 'Y':
        plt.savefig("../Pictures/3_DivIntoTwoClasses.png", dpi=200)
    if SHOW == 'Y':
        plt.show()