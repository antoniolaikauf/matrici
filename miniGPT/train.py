from prepare import train_data, val_data, vocab_size, n
import torch
from model import miniGPT , configGPT
import matplotlib.pyplot as plt

learning_rate = 6e-4 # max learning rate
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = 2000 # how many steps to warm up for

content_window = 8 # quantità token inseriti all'interno del modello
batch = 4
epoch = 1
amount_batch = n // batch # quantità totale di batch per ogni epoch

max_iters = 600000 # total number of training iterations
iter_num = 0 # numero attuale di iterazione

def get_batch(mode):
    if mode == 'train': data = train_data
    else: data = val_data
    id = torch.randint(len(data) - content_window,(batch,)) # range in cui prendere i token e quanti prenderne
    x = torch.stack([data[id_x:id_x + content_window] for id_x in id])
    y = torch.stack([data [id_y + 1:id_y + content_window + 1] for id_y in id])

    return x, y

# warmup learning rate e Cosine decay implementare
def get_lr(iteration):
    if iteration < warmup_iters: 
        return learning_rate * (iteration + 1) / (warmup_iters + 1)

m = miniGPT(configGPT)
num_parameters = m.get_params()
print(f"numero totale dei parametri del modello: {num_parameters}")
optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate, momentum=0.9)
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
x, y = get_batch("train")

while True:

    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loss, logit = m(x, y)

    print(f"step: {iter_num}, Loss: {loss}")

    loss_array.append(loss.data)
    x, y = get_batch("train")
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

# si prendono i logits dell ultimo token 
# last_token = logits[:, -1, :] 
# grazie alla softmax si ottiene la probabilita dell'ultimo token
# probability = F.softmax(last_token, dim=-1) 
# si scelgono i 50 token con la probabilità più alta
# token_prob, token_index = torch.topk(probability, 50, dim=-1) # forma: (batch, 50)
# si prende l'indice di un singolo token da quei 50 token (ognuno ha una probabilità distribuita per essere scelto)
# id_random_token_prob = torch.multinomial(token_prob, 1) # forma: (batch, 1)
# grazie all'indice si prende il token da token_prob  
# random_token_prob = torch.gather(token_prob, id_random_token_prob, dim=-1) # forma: (batch, 1)
# si attaccano i token alle proprie row 
# x = torch.cat((x, random_token_prob), dim=1) # forma: (batch, content_window + 1)