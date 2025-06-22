import os
import requests
import torch

file_path = os.path.join(os.path.dirname(__file__), 'text.txt')
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists(file_path):
    data_url = requests.get(url)
    with open(file_path, 'w') as f:
        f.write(data_url.text)

with open(file_path, 'r') as f:
    data = f.read()

chars = sorted(set(list(data)))
vocab_size = len(chars)

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