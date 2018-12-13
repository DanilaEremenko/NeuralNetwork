from __future__ import print_function
import LABS.ZeroLab.E_Function as dataset5

import keras
import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    models = [
        keras.models.load_model('C_3Adadelta_10000_0.01_80_40_20_linear/C_3Adadelta_10000_0.01.h5'),
        keras.models.load_model('C_Adadelta_10000_0.01_40_22_linear/C_Adadelta_10000_0.01.h5'),
        keras.models.load_model('C_Adadelta_10000_0.01_40_20_sigmoid/C_Adadelta_10000_0.01_40_20_1.h5'),
        keras.models.load_model('C_Adadelta_10000_0.01_60_30_linear/C_Adadelta_10000_0.01.h5')
    ]

    models_names = ["Adadelta_10000_0.01_80_40_20_linear",
                    "Adadelta_10000_0.01_40_20_sigmoid",
                    "Adadelta_10000_0.01_40_22_linear",
                    "Adadelta_10000_0.01_60_30_linear"
                    ]

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=2000, show=False)

    for model, model_name in zip(models, models_names):
        score = model.evaluate(x_test, y_test, verbose=0)
        plt.plot(np.transpose(x_test)[0], y_test)
        plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
        plt.title(model_name + "\naccuracy = %.4f" % score[0])
        plt.show()
