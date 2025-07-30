from prepare import train_data, val_data, n, decode
import torch
from torch.nn import functional as F
from model import miniGPT , configGPT
import matplotlib.pyplot as plt
import math

learning_rate = 6e-4 # max learning rate
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla

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
    # inizio training si alza il learning rate
    if iteration < warmup_iters:
         return (iteration + 1) * learning_rate / (warmup_iters + 1)
    elif (iteration > lr_decay_iters):
        return min_lr
    
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 1 + math.cos(math.pi * decay_ratio)
    # finito il warm up si cerca di abbassarlo cosi che non si abbia divergenza
    return min_lr + 0.5 * (learning_rate - min_lr) * coeff

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


'''
array_lr = []
for x in range(max_iters):
    prova = get_lr(x)
    array_lr.append(prova)
    print(prova)

print(len(array_lr))
plt.axis((0, max_iters, 0 ,max(array_lr)))
plt.xlabel('iteration')
plt.ylabel('lr')
plt.plot(array_lr)
plt.show()
'''

while True:
    # inferance ogni 250 step 
    if iter_num % 250 == 0:
        m.eval()
        with torch.no_grad():
            x, y = get_batch('')
            loss, logit = m(x)
            # calcolo probabilità su ultimi token di ogni row
            probabilityes = F.softmax(logit, dim=-1)
            # presa di solo 30 token con propbabilità più alta 
            token_prob, tokens_index = torch.topk(probabilityes, 30, dim=-1)
            # ottenimento di un token randomico
            token_index = torch.multinomial(tokens_index.float(), 1)
            # concatenazione del token con la frase 
            out = torch.cat((x , token_index), dim=1)

            for row in range(batch):
                sentence = decode(out[row].tolist())
                print(f"batch numero {batch}")
                print(f"token predetto --> {sentence[-1]}")
                print(f"frase {row} --> {sentence}")
                print("----------------------------")
            
    else:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch("train")
        loss, logit = m(x, y)

        print(f"step: {iter_num}, Loss: {loss}")

        loss_array.append(loss.data)
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