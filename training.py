from  MNIST.src import mnist_loader
from reteNeurale import Network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)
net = Network([784, 40, 10])

'''
più aumenti gli epoch e i neuroni nel layer e più l'accuratezza dovrebbe aumentare
'''
# net.SGD(40, 10, training_data, 3.0, test_data=None)
net.SGD(40, 10, training_data, 3.0, test_data=test_data)
