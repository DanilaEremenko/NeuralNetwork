import LABS.ZeroLab.F_ImagesGenerator as dataset8
from sklearn import metrics

from keras_preprocessing import image
from neupy import algorithms
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    pnn = algorithms.PNN(std=10,batch_size=128,verbose=True)

    pnn.train(x_train[0:10000], y_train[0:10000])

    y_predicted = pnn.predict(x_test)

    local_path = 'MY_IMAGES_GIMP/'
    for nTest in np.arange(1, 10, 1):
        img = image.load_img(local_path + str(nTest) + '.png', target_size=(28, 28), color_mode='grayscale')

        # convert to numpy array
        x = image.img_to_array(img)

        # Inverting and normalizing image
        x = 255 - x
        x /= 255
        x = np.expand_dims(x, axis=0)
        x.shape = (1, 784)

        prediction = pnn.predict(x)

        print('REAL \t\tPREDICTED')
        print(str(nTest) + '\t\t\t' + str(np.argmax(prediction)))

    print("accuracy = %.2f" % (metrics.accuracy_score(y_predicted,y_test)))
