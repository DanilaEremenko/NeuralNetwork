def func(x1: int, x2: int, x3: int, x4: int, x5: int):
    return x1 & x2 | x3 | x4 & x5


n = 0
print('number \t operands \t\t result')
for x1 in range(0, 2):
    for x2 in range(0, 2):
        for x3 in range(0, 2):
            for x4 in range(0, 2):
                for x5 in range(0, 2):
                    print(n, '\t\t', x1, x2, x3, x4, x5, '\t\t',func(x1, x2, x3, x4, x5))
                    n += 1
