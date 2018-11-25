import E_Flatness_Classificator_Data as dataset5

if __name__ == '__main__':
    train_size = 4000

    # func_type=[lin,n_lin]
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=True, k=1, b=0,
                                                              func_type='lin')
