import numpy as np
from keras import Sequential
from keras.layers import Dense
import keras.initializers as wi


# TODO
def plot_weights():
    return 0


if __name__ == '__main__':
    ne_num = 4
    in_num = 1
    seed = 42

    print("\n--------------Zeros--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num, kernel_initializer=wi.Zeros(), bias_initializer=wi.Zeros()))
    print(model.get_weights())

    print("\n--------------Ones--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num, kernel_initializer=wi.Ones(), bias_initializer=wi.Ones()))
    print(model.get_weights())

    print("\n--------------Constant--------------\n")
    model = Sequential()
    model.add(
        Dense(ne_num, input_dim=in_num, kernel_initializer=wi.Constant(value=2.0), bias_initializer=wi.Constant(value=2.0)))
    print(model.get_weights())

    print("\n--------------RandomNormal--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num, kernel_initializer=wi.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
                    bias_initializer=wi.RandomNormal(mean=0.0, stddev=0.05, seed=seed)))
    print(model.get_weights())

    print("\n--------------RandomUniform--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num, kernel_initializer=wi.RandomUniform(minval=-0.05, maxval=0.05, seed=seed),
                    bias_initializer=wi.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)))
    print(model.get_weights())

    print("\n--------------TruncatedNormal--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num, kernel_initializer=wi.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed),
                    bias_initializer=wi.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed)))
    print(model.get_weights())

    print("\n--------------VarianceScaling--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed),
                    bias_initializer=wi.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed)))
    print(model.get_weights())

    # Special architecture is necessary
    # print("\n--------------Orthogonal--------------\n")
    # model = Sequential()
    # model.add(Dense(ne_num, input_dim=1,
    #                 kernel_initializer=wi.Orthogonal(gain=1.0, seed=None),
    #                 bias_initializer=wi.Orthogonal(gain=1.0, seed=None)))
    # print(model.get_weights())
    #
    # print("\n--------------Identity--------------\n")
    # model = Sequential()
    # model.add(Dense(ne_num, input_dim=1,
    #                 kernel_initializer=wi.Identity(gain=1.0),
    #                 bias_initializer=wi.Identity(gain=1.0)))
    # print(model.get_weights())

    print("\n--------------lecun_normal--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.lecun_normal(seed=seed),
                    bias_initializer=wi.lecun_normal(seed=seed)))
    print(model.get_weights())

    print("\n--------------lecun_uniform--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.lecun_uniform(seed=seed),
                    bias_initializer=wi.lecun_uniform(seed=seed)))
    print(model.get_weights())

    print("\n--------------glorot_normal--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.glorot_normal(seed=seed),
                    bias_initializer=wi.glorot_normal(seed=seed)))
    print(model.get_weights())

    print("\n--------------glorot_uniform--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.glorot_uniform(seed=seed),
                    bias_initializer=wi.glorot_uniform(seed=seed)))
    print(model.get_weights())

    print("\n--------------he_normal--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.he_normal(seed=seed),
                    bias_initializer=wi.he_normal(seed=seed)))
    print(model.get_weights())

    print("\n--------------he_uniform--------------\n")
    model = Sequential()
    model.add(Dense(ne_num, input_dim=in_num,
                    kernel_initializer=wi.he_uniform(seed=seed),
                    bias_initializer=wi.he_uniform(seed=seed)))
    print(model.get_weights())
