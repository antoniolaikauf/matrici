import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

#--------------------------------
# DATASET
#--------------------------------

idx_train = idx[:int(N * 0.7)]
idx_val = idx[int(N * 0.7):]

train_x, train_y = x[idx_train], y[idx_train]

val_x, val_y = x[idx_val], y[idx_val] 

class CustomDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __getitem__(self, index):
        return train_x[index], train_y[index]
    
    def __len__(self):
        return len(self.train_x)

train_data = CustomDataset(train_x, train_y)

train_loader = DataLoader(train_data, 16, True)


def eval_mode(model, x, y):
    with torch.no_grad():
        model.eval()
        out = model(x)
        loss = loss_fn(out, y)

    return loss


#--------------------------------
# RETE NEURALE
#--------------------------------


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super(ManualLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

epoch = 40
lr = 0.01
loss_parameter = []
loss_eval = []

mlp = ManualLinearRegression().to(device)
parameters_model = list(mlp.parameters())
Optimizer = torch.optim.SGD(parameters_model, lr=lr)
loss_fn = nn.MSELoss(reduction='mean') # mean squared error  SSM sum squered error

for _ in range(epoch):
    loss_batch = []
    for x_batch, y_batch in train_loader:
        x_batch.to(device)
        y_batch.to(device)

        mlp.train()
        y_hat = mlp(x_batch)

        # print(list(mlp.parameters()))
        loss = loss_fn(y_hat, y_batch)
        loss.backward(retain_graph=True) # si inserisce retain_graph=True perchè cosi il grafo non viene perso
        loss_batch.append(loss.detach().numpy())
        Optimizer.step()
        Optimizer.zero_grad()

        # validazione 

    loss_eval.append(eval_mode(mlp, val_x, val_y))


    mean_loss = np.mean(loss_batch)
    # print(mean_loss)
    # print(loss_eval)
    loss_parameter.append(mean_loss) 
# make_dot(y).render("rnn_torchviz", format='png')
print(mlp.state_dict())
# plt.plot(loss_parameter)
# plt.plot(loss_eval)
# plt.show()


#------------------------------------
# SALVATAGGIO DI UN MODELLO
#------------------------------------


check_point = {
    'epoch' : epoch,
    'model_state_dict': mlp.state_dict(),
    'optimizer_state_dict': Optimizer.state_dict(),
    'loss': loss,
    'val_loss':loss_eval
}

torch.save(check_point, 'model_checkpoint.pth')

#-------------------------------------
# CARICAMENTO DI UN MODELLO
#-------------------------------------

model_check_point = torch.load('model_checkpoint.pth')

mlp.load_state_dict(model_check_point['model_state_dict'])
Optimizer.load_state_dict(model_check_point['optimizer_state_dict'])

saved_epoch = model_check_point['epoch']
saved_loss = model_check_point['loss']
saved_loss_eval = model_check_point['val_loss'] 

mlp.train()


dummy_labels = torch.tensor([1.0, 0.0])
dummy_predictions = torch.tensor([.9, .2])

negative_pred = dummy_predictions[dummy_labels == 0]
second_summation = torch.log(1 - negative_pred).sum()

print(negative_pred)


#-------------------------------------
# loss BCEWithLogitsLoss
#-------------------------------------
'''
quando un dataset è sbilanciato come in questo caso che le classi 
positive sono inferiori alle classi negative, la BCE si una pos_weight
che viene calcolata in base a la quantità di classi negative / le classi positive
e serve per far pesare di più la classe positva, incvoraggiando il modello 
a prestare attenzione a questa classe 
'''
loss_fn_BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

logit = torch.tensor([0.9, -0.4, 0.3])
y = torch.tensor([1.0, 0.0, 1.0 ])

loss_BCEWithLogitsLoss = loss_fn_BCEWithLogitsLoss(logit, y)

print(loss_BCEWithLogitsLoss)