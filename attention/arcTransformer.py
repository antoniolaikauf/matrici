from torch import nn # nn sta per rete neurale e serve per creare una rete neurale (ha all'interno tutti i componenti per la costruzione)
from torch.nn.functional import log_softmax
from torch import randn
# module sarebbe la classe con tutte le funzionalità per creare una rete neurale
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__(EncoderDecoder, self)
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
        super().__init__(Generator, self)
        self.proj = nn.Linear(d_model, vocab)

    def foward(self, x):
        return log_softmax(self.proj(x), dim = 1) # 2.71**x / sum(2.71**ogni valore della matrice)
    