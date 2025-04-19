#-------------------------
# input layer 784 perchè l'immagine è composta da 28x28 pixel
# in cui 0 rappresenterebbe bianco e 1 nero e i valori tra 0 e 1 
# # scaleranno di gradazione  
# un idden layer di 15 neuroni 
# un output layer di 10 neuroni in cui se si attiva il primo neurone
# allora la rete pensa che l'input sia il numero 0, se si illumina il 2
# allora la rete pensa che il numero sia il 1

import numpy as np
from numpy.random import default_rng
import random

class Network:
    # sizes rapresenterebbe i neuroni che appartengono ad ogni layer [2, 3, 4] questo ha 3 layer in cui il primo strato ha 2 neuroni il secondo 
    # 3 e il terzo ne ha 4  
    def __init__(self, sizes):
        self.size = sizes
        self.num_layer = len(sizes)
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]] # inizializzazione bias
        self.weights = [np.random.randn(sizes[x + 1], sizes[x]) for x in range(len(sizes) - 1)] # ializzazione weight

    def sigmod(self, Z):
        # print(Z)
        return 1 / (1 + np.exp(-Z))
    
    def feedFoward(self, input):
        '''
        # in feedfoward avviene la funzione che permette di ottenere l'output con una funzione lineare
        # che dopo viene usata come input per il prossimo strato 
        la dimensione dell'output di ogni stato deve essere uguale alla quantià di neuroni presenti per
        ogni strato
        '''

        for weightIdx , biasIdx in zip(self.weights, self.bias):
            output = self.sigmod(np.dot(weightIdx, input) + biasIdx)  
            print(f"prima {np.dot(weightIdx, input)}" )  
            print(f"weight usate {weightIdx}\nbias usata {biasIdx}\ninput usato {input} --> output ottenuto {output} grandezza output = {len(output)}\n")
            input = output
        return output
    
    def updateMiniBatch(self, batch, lr):
        # emptyB = [np.zeros((len(biasId), len(biasId[0]))) for biasId in self.bias]
        # emptyW = [np.zeros((len(weightId), len(weightId[0]))) for weightId in self.weights]
        gradB = [np.zeros(biasId.shape) for biasId in self.bias]
        gradW = [np.zeros(weightId.shape) for weightId in self.weights]
        print(gradB)
        print(gradW)
        self.weight = [w - ((lr / len(batch)) * newWeight) for w, newWeight in zip(self.weights, gradW) ]
        self.bias =  [b - ((lr / len(batch)) * newBias) for b, newBias in zip(self.bias, gradB)]
        

    def SGD(self, epochs, batchSize, trainingData, learningRate,  test_data=None ):
        '''
        The "training_data" è una lista di elementi accoppiati "(x, y)" che rappresenta l'input 
        e il desisderato output. Come primo procedimento si mescolano gli elementi dell'array essendo 
        ce dopo si prende da esso un batch di dimensioni ridotte rispetto al trainingData
        '''
        lenTraingData = len(trainingData)
        for epochId in range(epochs):
            random.shuffle(trainingData)
            miniBatchs = [trainingData[k:k + batchSize] for k in range(0, lenTraingData, batchSize)]
            for miniBatch in miniBatchs:
                self.updateMiniBatch(miniBatch, learningRate) 
            # print(f"epoch {epochId} completato")

    def lossFunction(self, trainingData):
        a = [dataId[0] - dataId[1] for dataId in trainingData]
        loss = 1 / 2 * sum(a)
        return loss





net = Network([2, 3, 4])
inputRete = default_rng(10).random((2,1))
# print(net.bias)
# print(net.weights)
print(net.feedFoward(inputRete))
# print(net.SGD(10, 3, [1,2,3,4,5,6,7,8,9], 0.01))
# net.updateMiniBatch(1,2)