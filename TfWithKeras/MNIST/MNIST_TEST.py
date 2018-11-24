import numpy as np
import keras

from keras.preprocessing import image

if __name__ == '__main__':
    local_path = 'MY_IMAGES_GIMP/'
    for nTest in np.arange(1, 10, 1):
        img = image.load_img(local_path + str(nTest) + '.png', target_size=(28, 28), grayscale=True)

        # convert to numpy array
        x = image.img_to_array(img)

        # Inverting and normalizing image
        x = 255 - x
        x /= 255
        x = np.expand_dims(x, axis=0)
        x.shape = (1, 784)

        model = keras.models.load_model('MNIST_MODEL.h5')

        prediction = model.predict(x)

        print('REAL \t\tPREDICTED')
        print(str(nTest) + '\t\t\t' + str(np.argmax(prediction)))
