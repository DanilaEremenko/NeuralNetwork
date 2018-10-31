import random
from math import fabs
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), 0)
ds.addSample((0, 1), 1)
ds.addSample((1, 0), 1)
ds.addSample((1, 1), 0)

ER_THRESHOLD = 1

networkNumber = 0
i = 0

while i < 2:

    networkNumber += 1
    print(networkNumber)

    hiddenLayerNumber = random.randint(0, 50)
    print('hiddenLayerNumber = ', hiddenLayerNumber)

    net = buildNetwork(2, hiddenLayerNumber, 1)

    trainer = BackpropTrainer(net, ds)

    trainer.trainUntilConvergence()

    currentError = fabs(net.activate([0, 0]) - 0.0)
    currentError += fabs(net.activate([0, 1]) - 1.0)
    currentError += fabs(net.activate([1, 0]) - 1.0)
    currentError += fabs(net.activate([1, 1]) - 0.0)
    print('currentError = ', currentError)

    if currentError < ER_THRESHOLD:
        print('Results:')
        print(net.activate([0, 0]))
        print(net.activate([0, 1]))
        print(net.activate([1, 0]))
        print(net.activate([1, 1]))
        i += 1

    print('-------------------')
