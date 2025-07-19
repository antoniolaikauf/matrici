from prepare import train_data, val_data, vocab_size, n
import torch
from model import miniGPT
import matplotlib.pyplot as plt

content_window = 8 # quantità token inseriti all'interno del modello
batch = 4
epoch = 1
amount_batch = n // batch # quantità totale di batch per ogni epoch

max_iters = 600000 # total number of training iterations
iter_num = 0 # numero attuale di iterazione

configGPT = {
    'n_head' : 8,
    'n_embd' : 512, 
    'vocab_size' : vocab_size,
    'n_layer' : 6,
    'contex_size': 8 
}

def get_batch(mode):
    if mode == 'train': data = train_data
    else: data = val_data
    id = torch.randint(len(data) - content_window,(batch,)) # range in cui prendere i token e quanti prenderne
    x = torch.stack([data[id_x:id_x + content_window] for id_x in id])
    y = torch.stack([data [id_y + 1:id_y + content_window + 1] for id_y in id])

    return x, y


m = miniGPT(configGPT)
optimizer = torch.optim.SGD(m.parameters(), lr=0.0003, momentum=0.9)
loss_array = []


# questo tipo di allenamento con epoch viene usato di solito con piccoli dataset
'''

for id_epoch in range(epoch):
    for id_batch in range(amount_batch):
        x, y = get_batch("train")
        loss, logits = m(x, y)
        loss.backward()
        loss_array.append(loss.data)
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        
    print(f"eseguito batch numero: {id_epoch}")
'''

# loop di allenamento

X, Y = get_batch("train")

while True:
    loss, logit = m(X, Y)
    print(f"step: {iter_num}, Loss: {loss}")
    loss_array.append(loss.data)
    X, Y = get_batch("train")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    iter_num += 1

    if iter_num > max_iters:
        break

plt.axis((0, amount_batch, 0, loss_array[0]))
plt.title("loss Function")
plt.xlabel("id_batch")
plt.ylabel("Loss")
plt.plot(loss_array)
plt.show()
# creare la mask dopo il prodotto scalare tra Q x K si ha una matrice T x T e si applica prima della softmax