from torch import nn
import torch
from prepare import vocab_size
import math
from torch.nn import functional as F

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
        assert config['d_model'] % config['n_head'] == 0
        # si calcolano le Q, K, V tramite una trasformazione lineare per migliorare la computazione/efficienza
        self.c_attn = nn.Linear(config['d_model'], config['d_model'] * 3)
        self.n_head = config['n_head']
        self.d_model = config['d_model']
        self.config = config
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, T, C = x.size() 
        # self.c_attn ha dimensioni B, T, C          B = batch dimesion  T = quantità di token  C = d_model * 3 che sarebbero le query key value
        # # si inserisce dim=2 perchè lo si vuole dividere le C dimension, se si volesse dividere per batch sarebbe dim=0  
        q ,k ,v = self.c_attn(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_q)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_v)

        att = (q @ k.transpose(2, 3)) / (math.sqrt(k.size(-1)))
        att = self.softmax(att) # softmax la si esegue sull'ultima dimensione
        y = att @ v

        # contiguous viene usata quando ci sono cambiamenti dell'organizzazione dei dati
        # pèerchè durante i cambiamenti dell'organizzazione dei dati questi vedi immagine contiguous.png per capire
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y




class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config['d_model'], config['d_model'] * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config['d_model'] * 4, config['d_model'])

    def forward(self, x):
        # eseguita prima funzione lineare --> eseguita RELU che fornisce la non linearità --> eseguita la seconda funzione lineare
        return self.linear2(self.relu(self.linear1(x)))
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config['d_model']) # si passa dentro la diomensione su cui si vuole fare la normalizzazione
        self.attention = Attention(config)
        self.layer_norm2 = nn.LayerNorm(config['d_model'])
        self.ffn = FFN(config)

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
        
        '''
        questo avviene solo durante l'inferenza e non durante l'addestramento 
        questa ultima linearizzazione è importante essendo che trasforma l'output che ora ha dimnsioni 
        (B, T, C) in (B, vocab_size) questo avviene tramite dei pesi.
        Da questi logits viene preso solo quello dell'ultimo token perchè è da quello
        che si predirrà il token successivo 
        es.
        vocabolario:
        0: "ciao"

        1: "come"

        2: "Mondo"

        3: "bene"

        4: "stai"

        la frase è 'ciao mondo come' e si deve prevedere il token successivo stai
        ora si ha un output di dimensioni (1, 3, 5) dopo aver fatto la linearizzazione 
        ma l'input da inserire dentro aalla softmax è solo quello dell'ultimo token che
        sarà [0.2, 2, 1.5, -1.3, 3] una volta uscitra dalla softmax sarà softmax([0.2, 2, 1.5, -1.3, 3]) ≈ [0.11, 0.12, 0.13, 0.14, 0.52]
        e quindi si prenderà 0.52 che rappresenta stai nel vocabolario
        '''
        
        self.linear = nn.Linear(config['d_model'], config['vocab_size']) 
        self.softmax = nn.Softmax(dim=-1)

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

    def forward(self, x, target):
        # si ottiene la dimensione del tensor
        batch, token = x.size()
        position_token = torch.arange(0, token, dtype=torch.long)
        # si inserisce i valori per ottenere i vettori dei token 
        token_embedding = self.transformer.w_token_embedding(x)
        # si inserisce le posizioni dei token 
        position_embedding = self.transformer.w_position_embedding(position_token)
        x = token_embedding + position_embedding

        for block in self.transformer['block']:
            x = block(x)
        
        logits = self.linear(x)  # Forma: (batch, token, vocab_size), cioè (B, T, C)
        B, T, C = logits.size()

        # la cross_entropy fa gia di suo internamente una softmax
        # last_token = logits[:,-1,:]
        # softmax = self.softmax(last_token)

        '''
        la loss viene calcolata per la predizione del token successivo dopo che si fa la softmax 
        e si calcolano la predizzione su vocab_size, quale token prendere viene dato da il target
        quindi se noi abbiamo elaborato un token 'ciao' e dobbiamo prevedere il target 'come'
        il token ciao avrà prodotto un vettore di probabilità [0.2, 0.3, 2]
        se 'come' ha indice 3 allora si prende 0.3 e si farà la formula di immagine Cross_entropy_loss_2 e Cross_entropy_loss
        '''
        loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return loss, logits
       

# m = miniGPT(configGPT)
# m("qua si passerà l'intero batch ")
# print(m(torch.randint(0, 6, (2,8))))

# print(m.get_params())
# print(configGPT['vocab_size'])
