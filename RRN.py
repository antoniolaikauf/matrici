

class Encoder:
    def __init__(self):
        pass

    def embedding(self, x):
        words = x.split(' ')
        

    def Foward(self, x):
        inputs = self.embedding(x)

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
        pass

    def decoder(self):
        pass




x = 'ciao come stai'

rrn = RRN(x)
rrn.encoder()
