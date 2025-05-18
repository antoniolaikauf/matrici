from Tokenizator import Tokenizer
from Embedding import Embedding

class Encoder:
    def __init__(self, d_model = 512, N = 6):
        self.d_model = d_model
        self.N = N
        pass

    def embedding(self, x):
        tokens = Tokenizer(x)
        wordEncoded = tokens.encode()
        embedding = Embedding(self.d_model, wordEncoded)
        wordsEmbedded = embedding.getCombinedEmbedding()
        # print(tokens.vocab)
        # print(tokens.decode(wordEncoded))
        return wordsEmbedded


    def Foward(self, x):
        inputs = self.embedding(x)
        return inputs

    def backward(self):
        pass


class  Decoder:
    def __init__(self):
        pass

    def foward(self):
        pass


class RRN:
    def __init__(self, x):
        self.x = x
        
    def encoder(self):
        h = Encoder()
        foward = h.Foward(self.x)
        return foward

    def decoder(self):
        pass


x = 'ciao ciao'

rrn = RRN(x)
print(rrn.encoder())
