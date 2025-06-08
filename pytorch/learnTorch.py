import torch
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot

zero_dimension = torch.tensor(3.14)
one_dimension = torch.tensor([3.14, 2, 6, 8])
two_dimension = torch.ones((2, 3))
three_dimension = torch.randn((2, 4, 3))

two_dimension = two_dimension.view((3,2))

# clone crea direttamente un nuovo tensor a differenza di view che modifica lo stesso tensore
copy_two_dimension = two_dimension.clone()
# print(hex(id(copy_two_dimension)))
# print(hex(id(two_dimension)))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 100

torch.manual_seed(42)

idx = np.arange(N)
x = torch.rand((N, 1))
w = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)
y = x * w + b

idx_train = idx[:int(N * 0.7)]
idx_val = idx[int(N * 0.7):]

train_x, train_y = x[idx_train], y[idx_train]
val_x, val_y = x[idx_val], y[idx_val] 

w1 = torch.randn(1, requires_grad=True, device=device)
b1 = torch.randn(1, requires_grad=True, device=device)


epoch = 100
lr = 0.01
loss_parameter = []

Optimizer = torch.optim.SGD([w1, b1], lr=lr)
loss_fn = torch.nn.MSELoss(reduction='mean') # mean squared error  SSM sum squered error

for _ in range(epoch):
    y_hat = train_x * w1 + b1
    loss = loss_fn(y_hat, train_y)
    loss.backward(retain_graph=True) # si inserisce retain_graph=True perchè cosi il grafo non viene perso
    loss_parameter.append(loss.detach().numpy())
    print(loss)
    Optimizer.step()
    Optimizer.zero_grad()

make_dot(y).render("rnn_torchviz", format='png')
# plt.plot(loss_parameter)
# plt.show()