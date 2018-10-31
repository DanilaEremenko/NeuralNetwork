from pybrain3.structure import FeedForwardNetwork

net = FeedForwardNetwork()

from pybrain3.structure import LinearLayer, SigmoidLayer

net.addInputModule(LinearLayer(2, name='in'))
net.addModule(SigmoidLayer(3, name='hidden'))
net.addOutputModule(LinearLayer(1, name='out'))

from pybrain3.structure import FullConnection

net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))

net.sortModules()

for i in range(0, 10):
    print(net.activate([1, 2]))
