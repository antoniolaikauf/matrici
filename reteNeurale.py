#-------------------------
# input layer 784 perchè l'immagine è composta da 28x28 pixel
# in cui 0 rappresenterebbe bianco e 1 nero e i valori tra 0 e 1 
# # scaleranno di gradazione  
# un idden layer di 15 neuroni 
# un output layer di 10 neuroni in cui se si attiva il primo neurone
# allora la rete pensa che l'input sia il numero 0, se si illumina il 2
# allora la rete pensa che il numero sia il 1

import numpy as np

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
        return 1 / (1 + np.exp(- Z))
    
    def feedFoward(self, input):
        
        for weightIdx , biasIdx in zip(self.weights, self.bias):
            output = self.sigmod(np.dot(weightIdx, input) + biasIdx)    
            print(f"weight usate {weightIdx}\nbias usata {biasIdx}\ninput usato {input} --> output ottenuto {output}\n")
            input = output
        return output

     
net = Network([2, 3, 4])
print(net.bias)
print(net.weights)
print(net.feedFoward([10, 20]))