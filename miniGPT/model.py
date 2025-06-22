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

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

class miniGPT(nn.Module):
    def __init__(self, config):
        super(miniGPT, self).__init__()
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
        moduleList 
        '''
        n_params_position_embedding = self.transformer['w_position_embedding'].weight.numel()
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= n_params_position_embedding
        return n_params

    def forward(self, x):
        pass
    
m = miniGPT(configGPT)

# print(m(torch.tensor(10)))
print(m.get_params())
print(configGPT['vocab_size'])