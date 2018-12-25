import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr
import TfWithKeras.CROSS_ZERO_RECOGNIZER.CrossZeroDataGen as dataset

if __name__ == '__main__':
    np.random.seed(42)
    in_image_size = (32, 32)
    ex_num = 10
    neur_number = 800
    example_dir = "A_CZ"

    # 0 - create data, 1 - load created data,2 - load created data and train
    MODE = 0

    if MODE == 0:
        # CREATE FORMATTED DIRECTORY
        templ_dir = "PICT_TEMPL/"
        x_pictures = np.array(
            [[templ_dir + "CircleV0.png", templ_dir + "CircleV1.png", templ_dir + "CircleV2.png",
              templ_dir + "CircleV3.png", templ_dir + "Circle_Gimp.png"],
             [templ_dir + "CrossV0.png", templ_dir + "CrossV1.png", templ_dir + "CrossV2.png",
              templ_dir + "CrossV3.png", templ_dir + "Cross_Gimp.png"]])
        y_types = np.array([0, 1])
        dataset.load_data_to_dir(ex_num, example_dir, images_size=in_image_size, x_pictures=x_pictures, y_types=y_types)
    elif MODE == 1 or MODE == 2:

        # LOAD DATA FROM FORMATTED DIRECTORY
        (x_train, y_train), (x_test, y_test) = dataset.load_data_from_dir(example_dir)

        if MODE == 2:
            x_train = x_train / 255.0

            model = Sequential()

            # <neur_number> neurons with 1024 inputs,initialize - normal distribution
            model.add(
                Dense(neur_number, input_dim=in_image_size[0] * in_image_size[1], init='normal', activation='relu'))
            model.add(Dense(1, init='normal', activation='hard_sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0008), metrics=['accuracy'])

            # batch_size define speed of studying
            history = model.fit(x_train, y_train, batch_size=1, nb_epoch=5, verbose=1)

            score = model.evaluate(x_test, y_test, verbose=1)
            print ("accuracy on testing data %.f%%" % (score[1] * 100))

            gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png", save=False, show=True)

            # model.save('CZ_REC_200.h5')
