'''
rete neurale senza librerie e con un solo neurone
'''

import matplotlib.pyplot as plt

data_set = [
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
]

epoch = 50
w = 2.1
eps = 0.01
lr = 0.001


def forward(data):
    output = [x[0] * w for x in data]
    return output

def lossFunction(y):
    loss = [(y[valueId] - data_set[valueId][1])**2 for valueId in range(len(y))]
    return sum(loss)

# RICORDARSI CHE I GRADIANTI SI FANNO SULLA LOSS FUNCTION E NON A QUELLA PER OTTENERE L'OUTPUT DELLA RETE 
def backPropagation(inputs, eps, w):
    output1 = [x[0] * w for x in inputs]
    loss1 = lossFunction(output1)

    # print(f'loss: {loss1}')
    output2 = [x[0] * (w + eps) for x in inputs]
    loss2 = lossFunction(output2)

    # print(f'loss: {loss2}')

    grad = (loss2 - loss1) / eps
    return grad

loss_value = []
for _ in range(epoch):
    print('\n---------------------------------')
    output = forward(data_set)
    loss = lossFunction(output)
    print(f'output rete neurale: {output}')
    grad = backPropagation(data_set, eps, w)
    w -= grad * lr
    loss_value.append(loss)

    print('---------------------------------')

output = forward(data_set)
print(f'OUTPUT FINALE: {output}')
print('W', w)
print(loss_value)

def plot():
    plt.axis((0, epoch, 0, loss_value[0],))
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(loss_value)
    plt.show()

plot()