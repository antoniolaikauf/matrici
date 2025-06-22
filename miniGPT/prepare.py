import os
import requests
import torch

from torch import nn

file_path = os.path.join(os.path.dirname(__file__), 'text.txt')
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists(file_path):
    data_url = requests.get(url)
    with open(file_path, 'w') as f:
        f.write(data_url.text)

with open(file_path, 'r') as f:
    data = f.read()


chars = sorted(set(list(data)))

char_to_int = { char:id_char for id_char, char in enumerate(chars) }
int_to_char = { id_char:char for id_char, char in enumerate(chars) }

def encode(text):
    return [char_to_int[char] for char in text]

def decode(tokens):
    return ''.join([int_to_char[token] for token in tokens])

n = len(data)
data = torch.tensor(encode(data), dtype=torch.long)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# quantità token 1115394
# quantità token per train 1003854
# quantità token per validazione 111540

content_window = 8
batch = 4
def get_batch(mode):
    if mode == 'train': data = train_data
    else: data = val_data
    id = torch.randint(len(data) - content_window,(batch,))
    x = torch.stack([data[id_x:id_x + content_window] for id_x in id])
    y = torch.stack([data [id_y + 1:id_y + content_window + 1] for id_y in id])

    return x, y

print(get_batch('train'))
class embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(embedding, self).__init__()
        self.embedding_table = nn.Embedding(vocab_size, d_model)

    def forward(self, idx):
        return self.embedding_table(idx)
    
vocab_size = len(chars)
d_model = 512
m = embedding(vocab_size, 512)

print(m(torch.tensor(10)))