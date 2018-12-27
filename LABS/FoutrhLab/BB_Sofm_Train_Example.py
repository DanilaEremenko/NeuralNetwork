import numpy as np
from neupy.exceptions import StopTraining

import matplotlib.pyplot as plt

def on_epoch_end(model):
    if model.train_errors.last() < goal_loss:
        raise StopTraining("Training has been interrupted")

def load_data(train_size=1000):
    x_train = np.random.rand((train_size, 2))



