import math
import numpy as np 
from numpy.random import default_rng

class GradiantDescent:
    def __init__(self):
        self.f = [0, 0, 1, 0]
        self.output = [0 ,0, 0, 0]
        self.hiddenLayer = [0.2, 0.4, -0.3, 0.05, 0.05]
        self.weight= default_rng(10).random((5,4))
        self.ln = 0.01
        self.bias = [-5, -3, 4, 6]
        self.n = len(self.hiddenLayer)

    def linearEquasion(self):
        for idx in range(len(self.output)):
            # XidxW * WidxW + Qidx 
            prod = [self.weight[idxW][idx] * self.hiddenLayer[idxW] for idxW in range(len(self.weight))]
            self.output[idx] = np.sum(prod) + self.bias[idx]
        return self.output 
        
    def lossFunction(self):
        # si calcola quanto sono differenti gli output dal l'ouput originale 
        calc = [round(math.pow((self.f[idx] - self.output[idx]), 2), 4) for idx in range(len(self.f))]
        return 1 / (2 * self.n) * sum(calc)

    def gradiantDescent(self):
        # creazione di matrici per gradientiW  e gradientiQ
        gradW = np.zeros((self.n, len(self.f)))
        gradQ = np.zeros(len(self.f))
        
        # queste sono le due derivate dall'equazione della loss function 

        # ∂C/∂Q = -1/m * ∑(f_j - output_j) rispetto a Q
        # ∂C/∂W = -1/m * ∑(f_j - output_j) * hiddenLayer_i rispetto a Q rispetto a W

        m = len(self.f)
        for idx in range(m):
            error = (self.f[idx] - self.output[idx])
            gradQ[idx] = error

            for i in range(self.n):
                gradW[i][idx] = error * self.hiddenLayer[i]

        # ∂C/∂Q
        gradQ = -gradQ / m
        # ∂C/∂W
        gradW = -gradW / m

        # ΔW = ∂C/∂W * ln
        # ΔQ = ∂C/∂Q * ln
        stepW = gradW * self.ln
        stepQ = gradQ * self.ln

        # aggiornamento dei parametri 
        self.weight -= stepW
        self.bias -= stepQ

        deltaC = np.sum(gradQ * stepQ) + np.sum(gradW * stepW)

        print(f"Gradienti pesi (∂C/∂W): {gradW}")
        print(f"Gradienti bias (∂C/∂Q): {gradQ}")
        print(f"Cambiamento totale della perdita (ΔC): {deltaC}")
        
        return round(deltaC, 6)
    
    def process(self):
        output = self.linearEquasion()
        loss =self.lossFunction()
        grad = self.gradiantDescent()
        print(f"loss attuale {loss} loss prevista {grad} miglioramento {loss - grad}")

G = GradiantDescent()

for x in range(10):
    G.process()



#-------------------------
# se si volesse usare la sigmod function allora busogna modificare la linearEquasion
# e anche la loss function 