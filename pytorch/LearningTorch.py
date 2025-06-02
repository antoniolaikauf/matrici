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

def foward():
    w1 = np.random.randn(1)
    b1 = np.random.randn(1)
    return w1 * train_x + b1
    
out = foward()
error = out - train_y

# calcolo della MSE mean square loss che consiste in 1/n sum(yhatn - yn)**
loss = (error**2).mean()
print(loss)


gradW = 2 * (train_x * error).mean() # mean sarebbe 1/len(train_x)
gradQ = 2 * error.mean()

print(gradW, gradQ)

# plt.plot(train_x, train_y, 'bo')
# plt.show()

w_random = np.linspace(w - 3, w + 3, 200)
b_random = np.linspace(b - 3, b + 3, 200)

# print(w_random)

'''
grafico che mostra la curva della loss se si tenesse fisso un parametro.
mostra come gli altri parametri influenzerebbero la curva
nel esempio sotto si tiene fisso il parametro delle bias 65
nel grafico mostra la curva della loss functione bisogna vedere se 
modificando le w la loss scende molto veloce (cosa molto positiva)
o scende lentamente (cosa positiva essendo che scende ma molto lenta)
'''

def curveGrad(w_parameter, b_parameter, data):
    tot_loss = []
    for testId in range(len(w_parameter)):
        out = w_parameter[testId] * data + b_parameter[65]
        error = out - train_y[0]
        loss = (error**2).mean()
        tot_loss.append(loss)

    print(tot_loss)
    plt.ylabel('MSE')
    plt.xlabel('w')

    # sdi prendono 10 valori
    num_ticks = 10
    space = len(w_parameter) // num_ticks
    # tick_positions = np.linspace(0, len(w_random) - 1, num_ticks, dtype=int)
    tick_positions = [ space * index for index in range(num_ticks)]

    print(tick_positions)
    # si estraggono i valori date dalle posizioni dei 10 valori di  tick_positions  
    plt.xticks(tick_positions, np.round(w_random[tick_positions], 2))
    plt.plot(tot_loss, linestyle='dashed')
    plt.show()

curveGrad(w_random, b_random, train_x.reshape(-1, 1, 1))