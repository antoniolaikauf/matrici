import numpy as np
import matplotlib.pyplot as plt
import math

N = 100
w = 1
b = 2
w1 = np.random.randn(1)
b1 = np.random.randn(1)
lr = 0.03
eps = np.random.randn(N, 1) * 0.1 # un alto rumore potrebbe dare fastidio al modello rendendogli difficoltàa imparare


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


x = np.random.rand(N, 1)
# x = test(x)
y = x*w + b + eps


#---------------------------
# DATASET
#---------------------------


idx = np.arange(N)
np.random.shuffle(idx)

id_train = idx[:int(N * 0.7)]
id_val = idx[int(N * 0.7):]

train_x, train_y = x[id_train], y[id_train] # input e output per il trainig 
val_x = x[id_val], y[id_val] # input e output per vedere come performa il modello dopo essere allenato


#---------------------------
# NET
#---------------------------


def foward(w1, b1, train_x):
    return w1 * train_x + b1

# def Backpropagation(output):
#     error = output - train_y
#     # calcolo della MSE mean square loss che consiste in 1/n sum(yhatn - yn)**
#     loss = np.mean(error**2)
#     gradW = 2 * np.mean(train_x * error) # mean sarebbe 1/len(train_x)
#     gradQ = 2 * np.mean(error)

#     return gradW, gradQ, loss

def mini_batch(output, batch_id, data_x):
    error = output - train_y[batch_id:batch_id + 10]
    # calcolo della MSE mean square loss che consiste in 1/n sum(yhatn - yn)**
    loss = np.mean(error**2)
    gradW = 2 * np.mean(data_x * error) # mean sarebbe 1/len(train_x)
    gradQ = 2 * np.mean(error)

    return gradW, gradQ, loss

# # processo di allenamento della rete 
# for epoch in range(7000):

#     # print(w1, b1)
#     out = foward(w1, b1, train_x)
#     grad_w1, grad_b1, loss = Backpropagation(out)
#     w1 -= lr * grad_w1
#     b1 -= lr * grad_b1
#     print(f"Epoch {epoch}, Batch {batch_id}, Loss: {loss}, w1: {w1[0]}, b1: {b1[0]}")

for epoch in range(80):

    # print(w1, b1)
    for batch_id in range(0, int(N * 0.7), 10):
        out = foward(w1, b1, train_x[batch_id:batch_id+10])
        grad_w1, grad_b1, loss = mini_batch(out, batch_id, train_x[batch_id:batch_id+10])
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        print(f"Epoch {epoch}, Batch {batch_id}, Loss: {loss}, w1: {w1[0]}, b1: {b1[0]}")

out = foward(w1, b1, train_x)
print(out[:10])
print(train_y[:10])
print(w1, b1)


'''
la Standardizing viene usata dopo la divisione dei dati e migliora le performance
della rete neurale perchè tutti i valori hanno una scala molto simile migliorando
anche la performance dei gradiant descent
se per esempio abbiamo due features come età e salario l'età può variare tra 0 e 100 anni 
invece il salario può variare da 500 a 5000 euro, con la Standardizing non esiste più
questa disparità tra eta e salario in termini di valore essendo che ora variano tutti
tra es 0 e 2500
'''

def standardize(train):
    media = np.mean(train)
    Standardizing = np.sqrt(np.power(train - media, 2).mean())
    scale = (train - media) / Standardizing
    return scale

# print(standardize(train_x))

w_random = np.linspace(w - 3, w + 3, 200)
b_random = np.linspace(b - 3, b + 3, 200)

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
        out = w_parameter[testId] * data + b_parameter
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

b_fix = b_random[65]
# curveGrad(w_random, b_fix, train_x.reshape(-1, 1, 1))