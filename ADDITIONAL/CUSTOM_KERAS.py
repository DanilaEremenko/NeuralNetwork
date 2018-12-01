import keras.backend as K


def hard_lim(x):
    return K.cast(K.greater_equal(x, 0), K.floatx())

