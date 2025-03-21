import numpy as np
import math

class token:
    def __init__(self):
        self.Token = 'hello'
        self.distribution = math.sqrt(6/256)

    def tokenInput(self):
        return [ord(char) for char in self.Token]
    
    def tokenInputVector(self):
        return {tk : np.full(256, self.distribution) for tk in self.tokenInput()}
        
class Transformers(token):
    def __init__(self):
        super().__init__()

    def tr(self):
        print(self.distribution)
        print(self.tokenInputVector())

t =Transformers()
print(t.tr())
        