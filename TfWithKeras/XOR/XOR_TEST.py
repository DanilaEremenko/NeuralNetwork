import numpy as np
import keras

if __name__ == '__main__':
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    model = keras.models.load_model("XOR_MODEL.h5")
    print(model.predict(inputs))