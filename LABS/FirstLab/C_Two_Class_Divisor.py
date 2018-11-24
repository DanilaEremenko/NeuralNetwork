import LABS.ZeroLab.Programms.C_DivIntoTwoClasses as dataset3

if __name__ == '__main__':
    train_size = 4000


    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=True)
