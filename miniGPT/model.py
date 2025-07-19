from torch import nn
import torch
from prepare import vocab_size
import math
from torch.nn import functional as F

configGPT = {
    'n_head' : 8,
    'n_embd' : 512, 
    'vocab_size' : vocab_size,
    'n_layer' : 6,
    'contex_size': 8 
}

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # si calcolano le Q, K, V tramite una trasformazione lineare per migliorare la computazione/efficienza
        self.c_attn = nn.Linear(config['n_embd'], config['n_embd'] * 3)
        # i pesi vengono scalati in base a quante residual connection si ha 
        self.c_attn.weight.data = self.c_attn.weight.data * (1 / math.sqrt(config['n_layer'] * 2))
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.config = config
        self.softmax = nn.Softmax(dim=-1) # softmax la si esegue sull'ultima dimensione
        
    def forward(self, x):
        B, T, C = x.size()
        # self.c_attn ha dimensioni B, T, C          B = batch dimesion  T = quantità di token  C = n_embd * 3 che sarebbero le query key value
        # # si inserisce dim=2 perchè lo si vuole dividere le C dimension, se si volesse dividere per batch sarebbe dim=0  
        q ,k ,v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_q)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, head, token, d_v)

        att = (q @ k.transpose(2, 3)) / (math.sqrt(k.size(-1))) # forma: (batch, n_head, token, token)
        '''
        0 è la diagonale principale  se si volesse far si che non passasse per la diagonale principale allora si modifica quel parametro
        es. diagonal = 0 
        [1,0,0]
        [1,1,0]
        [1,1,1]
        es. diagonal = 1
        [1,1,0]
        [1,1,1]
        [1,1,1]
        perchè 'sbloccherebbe' il valore della prima row e posizione 1 e man mano aumenta quel valore in 2, 3 e cosi via
        ''' 
        # creare la mask dopo il prodotto scalare tra Q x K si ha una matrice T x T e si applica prima della softmax
        mask = torch.tril(torch.ones(T, T), diagonal=0).bool()
        
        # Applicazione della maschera ai punteggi di attenzione ovunque si ha nella maschera False allora si mette -inf
        att = att.masked_fill(mask == False, -float('inf'))
        # print(att[0][0])
        att = self.softmax(att) # forma : (batch, n_head, token, token)
        y = att @ v
        
        '''
        y forma : (batch, n_head, token, d_v) 

        [1_att * v_1, 1_att * v_2 ... 1_att * v_64] 
        [2_att * v_1, 2_att * v_2 ... 2_att * v_64]
        ...
        [8_att * v_1, 8_att * v_2 ... 8_att * v_64] 
        '''

        # contiguous viene usata quando ci sono cambiamenti dell'organizzazione dei dati
        # pèerchè durante i cambiamenti dell'organizzazione dei dati questi vedi immagine contiguous.png per capire
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config['n_embd'], config['n_embd'] * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config['n_embd'] * 4, config['n_embd'])
        self.linear1.weight.data = self.linear1.weight.data * (1 / math.sqrt(config['n_layer'] * 2))
        self.linear2.weight.data = self.linear2.weight.data * (1 / math.sqrt(config['n_layer'] * 2))

    def forward(self, x):
        # eseguita prima funzione lineare --> eseguita RELU che fornisce la non linearità --> eseguita la seconda funzione lineare
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # nel layer normalizzazione quando si normalizza il tensor x il risulato non cambia 
        self.layer_norm1 = nn.LayerNorm(config['n_embd']) # si passa dentro la dimensione su cui si vuole fare la normalizzazione
        self.attention = Attention(config)
        self.layer_norm2 = nn.LayerNorm(config['n_embd'])
        self.ffn = FFN(config)

    def forward(self, x):
        # in questo caso qua si sta facendo una pre-LN che consiste di normalizzare gli input prima di passarli dentro al sublayer
        # nel paper attention is all you need si usa una post-LN 
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))

        return x
class miniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            w_token_embedding = nn.Embedding(config['vocab_size'], config['n_embd']),
            w_position_embedding = nn.Embedding(config['contex_size'], config['n_embd']), # non si sta usando la sinusoide position encoding ma si sta usando la absolute position encoding (apprendibile)
            # h = hidden
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            # qua nel paper hanno messo un altro layer normalization
            ln_f = nn.LayerNorm(config['n_embd']),
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

        la frase è 'ciao mondo come' e si deve prevedere il token successivo 'stai'
        ora si ha un output di dimensioni (1, 3, 5) dopo aver fatto la linearizzazione.
        L'input da inserire dentro alla softmax è solo quello dell'ultimo token che
        sarà [0.2, 2, 1.5, -1.3, 3] una volta uscita dalla softmax sarà softmax([0.2, 2, 1.5, -1.3, 3]) ≈ [0.11, 0.12, 0.13, 0.14, 0.52]
        e quindi si prenderà 0.52 che rappresenta 'stai' nel vocabolario
        '''
        
        self.linear = nn.Linear(config['n_embd'], config['vocab_size'], bias=False) 
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

        for block in self.transformer.h:
            x = block(x)
        
        logits = self.transformer.ln_f(x)

        logits = self.linear(x)  # Forma: (batch, token, vocab_size), cioè (B, T, C)
        # Forma: (batch, token, vocab_size) si ha questa forma essendo che si deve prevedere il token successivo su tutti i possibili token 
        B, T, C = logits.size()

        # last_token = logits[:,-1,:]
        # softmax = self.softmax(last_token)

        '''
        la loss viene calcolata per la predizione del token successivo dopo che si fa la softmax 
        e si calcolano la predizzione su vocab_size, quale token prendere viene dato da il target
        quindi se noi abbiamo elaborato un token 'ciao' e dobbiamo prevedere il target 'come'
        il token ciao avrà prodotto un vettore di probabilità [0.2, 0.3, 2]
        se 'come' ha indice 1 allora si prende 0.3 e si farà la formula di immagine Cross_entropy_loss_2 e Cross_entropy_loss
        '''

        # la cross_entropy fa gia di suo internamente una softmax
        loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return loss, logits
       

# m = miniGPT(configGPT)
# m("qua si passerà l'intero batch ")
# print(m(torch.randint(0, 6, (2,8))))

# print(m.get_params())
# print(configGPT['vocab_size'])
