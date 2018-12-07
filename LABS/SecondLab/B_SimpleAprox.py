from LABS.FirstLab.E_Flatness_Classificator_Data import load_data_func

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data_func(train_size=100, show=True, k=1, b=1, func_type='n_lin')
