from prepare import train_data, val_data, vocab_size
import torch
from model import miniGPT

configGPT = {
    'n_head' : 8,
    'n_embd' : 512, 
    'vocab_size' : vocab_size,
    'n_layer' : 6,
    'contex_size': 8 
}

content_window = 8
batch = 4

def get_batch(mode):
    if mode == 'train': data = train_data
    else: data = val_data
    id = torch.randint(len(data) - content_window,(batch,)) # range in cui prendere i token e quanti prenderne
    x = torch.stack([data[id_x:id_x + content_window] for id_x in id])
    y = torch.stack([data [id_y + 1:id_y + content_window + 1] for id_y in id])

    return x, y


m = miniGPT(configGPT)
optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)

for x in range(1):
    optimizer.zero_grad()
    x, y = get_batch('train')
    loss, logits = m(x, y)
    loss.backward()
    optimizer.step()
    print(loss)

# creare la mask dopo il prodotto scalare tra Q x K si ha una matrice T x T e si applica prima della softmax