import LABS.ZeroLab.E_Function as dataset5
from ADDITIONAL.IMPLEMENTATIONS.RBF.RBFN import RBFN

import matplotlib.pyplot as plt


import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)

    goal_loss=0.007
    for neu_num in np.arange(10, 60, step=5):
        model = RBFN(hidden_shape=neu_num, sigma=0.5)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = (np.abs(y_test - y_pred)).mean()

        plt.plot(x_test, y_test, 'b.', label='real')
        plt.plot(x_test, y_pred, 'r.', label='fit')
        plt.legend(loc='upper right')
        plt.title('RBFN from git\nneu_num = %.d\nmae = %.4f' %
                  (neu_num, mae))
        plt.show()

        if mae < goal_loss:
            break

