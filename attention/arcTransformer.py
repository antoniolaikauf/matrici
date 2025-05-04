import torch
from torch import nn # nn sta per rete neurale e serve per creare una rete neurale (ha all'interno tutti i componenti per la costruzione)
from torch.nn.functional import log_softmax
from torch import randn
import copy
import math
# module sarebbe la classe con tutte le funzionalità per creare una rete neurale
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # matrice dei vettori dei token dopo aver fatto l'embedding
        self.tgt_embed = tgt_embed # matrice dei vettori dei token output dopo aver fatto l'embedding
        self.generator = generator

    def foward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask) 
    
    def encode(self, src,  src_mask):
        self.encoder(self.src_embed, src, src_mask)

    def decode(self, memory, src_masck, tgt, tgt_mask):
        self.decoder(self.tgt_embed(tgt), memory, src_masck, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def foward(self, x):
        return log_softmax(self.proj(x), dim = 1) # 2.71**x / sum(2.71**ogni valore della matrice)
    

def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, Layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(Layer, N)
        self.norm = LayerNorm(Layer.size)

    '''
     la mask serve per aggiungere token nulli fino a che non si raggiunga la lunghezza decisa dall'inizio
     es se ogni batch deve avere 10 token ma una sequenza è troppo corta allora si aggiungono dei token PAD 
     fino ad arrivare alla lunghezza del batch e la mask sarebbe la rappresentazione dei token 
     se è un token normale allora avrà 1 se invece è un token PAD allora sarà 0 questo perchè non si vuole che 
     i token pad influenzino gli altri token 
    '''

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = size
        self.eps = eps
        self.a = nn.Parameter(torch.ones(self.size))
        self.b = nn.Parameter(torch.zeros(self.size))

    def forward(self, x):
        '''
         self.a * norm non è una moltiplicazioni tra matrici ma ad ogni row di norm viene motiplicato self.a ovviamente la 
         size deve combaciare 
         e norm + self.b fa la stessa cosa ma con l'addizione  

         con media viene calcolata la media di ogni token all'interno di x e la calcola su d_model essendo che c'è -1
         e la struttura di x è composta come x: [batch_size, seq_len, d_model]
         si fa la stessa cosa con varianza 
        '''
        
        Media = x.mean(dim=-1, keepdim=True)  
        Varianza = ((x - Media) ** 2).mean(dim=-1, keepdim=True) 
        Norm = (x - Media) / (torch.sqrt(Varianza) + self.eps)  # Normalizzazione
        return self.a * Norm + self.b  


