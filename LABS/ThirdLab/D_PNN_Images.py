import LABS.ZeroLab.F_ImagesGenerator as dataset8
from sklearn import metrics

from keras_preprocessing import image
from neupy import algorithms
import numpy as np
from ADDITIONAL.IMAGE_CHANGER import show_image_by_pxs, get_pxs, noise

def test_with_noise():
    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    pnn = algorithms.PNN(std=1, batch_size=128, verbose=True)

    pnn.train(x_train[0:10000], y_train[0:10000])

    y_predicted = pnn.predict(x_test)

    local_path = 'MY_IMAGES_GIMP/'
    for nTest in np.arange(0, 10, 1):
        # convert to numpy array
        x = get_pxs(local_path + str(nTest) + '.png')

        # Inverting and normalizing image
        x = 255 - x
        x /= 255
        x = np.expand_dims(x, axis=0)
        x.shape = (1, 784)

        prediction = pnn.predict(x)

        print('REAL \t\tPREDICTED')
        print(str(nTest) + '\t\t\t' + str(prediction[0]))

    print("accuracy = %.2f" % (metrics.accuracy_score(y_predicted, y_test)))


    for i in (0, 9999):
        x_test[i] = noise(x_test[i], 500)

    y_predicted = pnn.predict(x_test)

    print("accuracy on noise data = %.2f" % (metrics.accuracy_score(y_predicted, y_test)))

def diff_train_size():
    maes = [0, 0, 0, 0]
    train_size = [15000, 10000, 5000, 2000]
    std = [2, 1, 0.5, 0.25]

    for j in range(0, 4):
        (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

        pnn = algorithms.PNN(std=std[j], batch_size=128, verbose=True)

        pnn.train(x_train[0:train_size[j]], y_train[0:train_size[j]])

        y_predicted = pnn.predict(x_test)

        maes[j] = metrics.accuracy_score(y_predicted, y_test)

        print("accuracy = %.2f" % (maes[j]))

if __name__ == '__main__':
    test_with_noise()
    # diff_train_size()
