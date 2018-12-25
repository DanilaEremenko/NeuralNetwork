import LABS.ZeroLab.F_ImagesGenerator as dataset8
from sklearn import metrics

from keras_preprocessing import image
from neupy import algorithms
import numpy as np
from ADDITIONAL.IMAGE_CHANGER import show_image_by_pxs,get_pxs

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    pnn = algorithms.PNN(std=1,batch_size=128,verbose=True)

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

    print("accuracy = %.2f" % (metrics.accuracy_score(y_predicted,y_test)))
