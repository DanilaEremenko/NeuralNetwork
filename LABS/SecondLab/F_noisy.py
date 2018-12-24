import keras
from keras_preprocessing import image

import LABS.ZeroLab.F_ImagesGenerator as dataset8

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adadelta

import ADDITIONAL.GUI_REPORTER as gr
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal
from ADDITIONAL.IMAGE_CHANGER import deform_image, noise

if __name__ == '__main__':
    np.random.seed(42)
    # 1,2 initializing
    epochs = 10
    verbose = 1
    neurons_number = [128, 10]

    opt_name = "Adam"
    optimizer = Adam()

    goal_loss = 0.013

    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=1)

    model = Sequential()

    model.add(Dense(neurons_number[0], input_dim=784, activation='relu'))

    model.add(Dense(neurons_number[1], activation='softmax'))

    # 3 setting stopper
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x=x_train, y=y_train, epochs=epochs,
                        verbose=verbose, batch_size=128, callbacks=callbacks, validation_data=(x_test, y_test))

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save=False, show=True)

    score = model.evaluate(x_test, y_test, verbose=1)
    print("accuracy on testing data %.f%%" % (score[1] * 100))

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

        prediction = model.predict(x)

        print('REAL \t\tPREDICTED')
        print(str(nTest) + '\t\t\t' + str(np.argmax(prediction)))
