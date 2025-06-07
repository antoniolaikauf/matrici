import torch
import numpy as np

zero_dimension = torch.tensor(3.14)
one_dimension = torch.tensor([3.14, 2, 6, 8])
two_dimension = torch.ones((2, 3))
three_dimension = torch.randn((2, 4, 3))

two_dimension = two_dimension.view((3,2))

# clone crea direttamente un nuovo tensor a differenza di view che modifica lo stesso tensore
copy_two_dimension = two_dimension.clone()
# print(hex(id(copy_two_dimension)))
# print(hex(id(two_dimension)))

# print(zero_dimension, zero_dimension.size())
# print(one_dimension, one_dimension.size())
# print(two_dimension, two_dimension.size())
# print(three_dimension, three_dimension.size())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 100

idx = np.arange(N)
x = torch.rand((N, 1))
w = torch.randn(1)
b = torch.randn(1)
y = x * w + b

idx_train = idx[:int(N * 0.7)]
idx_val = idx[int(N * 0.7):]

train_x, train_y = x[idx_train], y[idx_train]
val_x, val_y = x[idx_val], y[idx_val] 
# print(y)

print(device)
y = y.to(device)
print(y)
