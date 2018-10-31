from pybrain3.structure import RecurrentNetwork
from pybrain3.structure import LinearLayer
from pybrain3.structure import SigmoidLayer
from pybrain3.structure import FullConnection

net = RecurrentNetwork()

net.addInputModule(LinearLayer(2, name='in'))
net.addModule(SigmoidLayer(3, name='hidden'))
net.addOutputModule(LinearLayer(1, name='out'))

net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))

net.addRecurrentConnection(FullConnection(net['hidden'], net['out'], name='c3'))

net.sortModules()

for i in range(0, 10):
    print(net.activate([2, 2]))
