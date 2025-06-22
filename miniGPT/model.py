from torch import nn
import torch
from prepare import vocab_size

configGPT = {
    'n_head' : 8,
    'd_model' : 512, 
    'vocab_size' : vocab_size,
    'n_layer' : 6,
    'contex_window': 8 
}

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # si calcolano le Q, K, V tramite una trasformazione lineare per migliorare la computazione/efficienza
        self.c_attn = nn.Linear(config['d_model'], config['d_model'] * 3)
        self.n_head = config['n_head']
        self.d_model = config['d_model']
        
    def forward(self, x):
        # self.c_attn ha dimensioni B, T, C          B = batch dimesion  T = quantità di token  C = d_model * 3 che sarebbero le query key value
        # # si inserisce dim=2 perchè lo si vuole dividere le C dimension, se si volesse dividere per batch sarebbe dim=0  
        q ,k ,v = self.c_attn(x).split(self.d_model, dim=2)

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()

class FFN(nn.Module):
    def __init__(self):
        super().__init__()

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNormalization()
        self.attention = Attention(config)
        self.layer_norm2 = LayerNormalization()
        self.ffn = FFN

    def forward(self, x):
        # in questo caso qua si sta facendo una pre-LN che consiste di normalizzare gli input prima di passarli dentro al sublayer
        # nel paper attention is all you need si us auna post-LN 
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))

        return x
class miniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            w_token_embedding = nn.Embedding(config['vocab_size'], config['d_model']),
            w_position_embedding = nn.Embedding(config['contex_window'], config['d_model']),
            block = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        ))

    def get_params(self):

        '''
        self.parameters() raccoglie tutti i parametri ricursivamente del modello 
        inclusi anche quelli dei blocchi essendo che sono all'interno di un 
        moduleListche permette di creare un submodule, invece classi come layerNorm 
        sono gia dei submodule e quindi li calcola automaticamente
        '''

        n_params_position_embedding = self.transformer['w_position_embedding'].weight.numel()
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= n_params_position_embedding
        return n_params

    def forward(self, x):
        pass


m = miniGPT(configGPT)
m("qua si passerà l'intero batch ")
print(m(torch.tensor(10)))
print(m.get_params())
print(configGPT['vocab_size'])
