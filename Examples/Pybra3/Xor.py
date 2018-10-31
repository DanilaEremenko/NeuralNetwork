from pybrain3.datasets import SupervisedDataSet
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), 0)
ds.addSample((0, 1), 1)
ds.addSample((1, 0), 1)
ds.addSample((1, 1), 0)

net = buildNetwork(2, 3, 1)
trainer = BackpropTrainer(net, ds)

print(trainer.trainUntilConvergence())

print(net.activate([0,0]))
print(net.activate([1,1]))
print(net.activate([0,1]))
print(net.activate([1,0]))


