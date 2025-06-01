import numpy as np
import matplotlib.pyplot as plt
from  numpy.random import rand, randn

N = 100
w = 1
b = 2
eps = randn(N, 1)

'''
si può vedere che con questa funzione e senza eps (epsilon)
l adistribuzione dei dati è troppo uniforme e non rappresenterebbe 
il mondo reale essendo che neò raccogliere datri c'è sempre un errore
inoltre eps che aggiunge rumore rende il dataset più complesso
cosi che il modello deve apprendere relazioni più complesse

'''
def test(x):
    for xId in range(len(x)):
        for yId in range(len(x) - xId - 1):
            if x[yId][0] > x[yId + 1][0]:
                change = x[yId][0]
                x[yId][0] = x[yId + 1][0]
                x[yId + 1][0] = change
    return x


x = rand(N, 1)
# x = test(x)
y = x*w + b + eps

idx = np.arange(N)
np.random.shuffle(idx)

id_train = idx[:int(N * 0.7)]
id_val = idx[int(N * 0.7):]

train_x, train_y = x[id_train], y[id_train] # input e output per il trainig 
val_x = x[id_val], y[id_val] # input e output per vedere come performa il modello dopo essere allenato
    
plt.plot(train_x, train_y, 'bo')
plt.show()