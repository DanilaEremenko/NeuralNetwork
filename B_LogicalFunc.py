import matplotlib.pyplot as plt


def func(x1: int, x2: int, x3: int, x4: int, x5: int):
    return x1 & x2 | x3 | x4 & x5


if __name__ == '__main__':

    listX1 = []
    listX2 = []
    listY1 = []
    listY2 = []

    n = 1
    print('number \t operands \t\t result')
    for x1 in range(0, 2):
        for x2 in range(0, 2):
            for x3 in range(0, 2):
                for x4 in range(0, 2):
                    for x5 in range(0, 2):
                        result = func(x1, x2, x3, x4, x5)
                        if (result):
                            # listX1.append(result)
                            listX1.append(n)
                            listY1.append(n)
                        else:
                            # listX2.append(result)
                            listX2.append(n)
                            listY2.append(n)

                        print(n, '\t\t', x1, x2, x3, x4, x5, '\t\t', func(x1, x2, x3, x4, x5))
                        n += 1

    plt.plot(listX1, listY1, '.b')
    plt.plot(listX2, listY2, '.r')

    plt.show()
