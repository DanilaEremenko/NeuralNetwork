import numpy as np

points = np.array([(2, 5), (5, 5), (5, 10)])
points_target = np.array([(5, 8), (5, 5), (10, 5)])


#X = Ax + By + C
#Y = Dx + Ey + F
#Matrix
#[X] = [A B C]      [x]
#[Y] = [D E F]   X  [y]
#[1] = [0 0 1]      [1]
for x, y in points:
    print(str(x) + str(y))

