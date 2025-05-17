import math
import numpy as np

'''
d_model sarebbe la grandezza dei vettori con cui si lavora all'interno del modello 

vocabSize sarebbe l grandezza di tutti i token possibili del modello 

maxToken sarebbe la grandezza di quanti token può gestire in una sola 
volta il modello 

tokenEmbedding sarebbe lo spazio vettoriale in cui i token (maxToken) sono 
rappresentati, qua parole con significato simile saranno molto vicine 
rispetto a parole che hanno sigificato completamente diverso

es.
cane avrà un vettore molto simile a quello di gatto  rispetto al vettore
della parola tavolo e quindi nello spazio vettoriale la parola cane sarà  
vicino alla parola gatto 
'''

class Embedding:
    def __init__(self, d_model, input, vocabSize = 30000, maxToken = 512):
        self.d_model = d_model
        self.input = input
        self.vocabSize = vocabSize
        self.maxToken = maxToken
        self.tokenEmbedding = np.random.randn(vocabSize, d_model) 
        self.positionalEmbedding = self.createPositionalEncoding()

    def createPositionalEncoding(self):

        '''
        createPositionalEncoding permette di creare una matrice di vettori 
        delle posizioni dei token all'interno della frase. Crea una matrice di 
        dimensione (MaxToken, d_model) ma restituisce solo una parte di essa che sarebbe
        quella con solo le posizioni dei token 
        nel paper attention is all you need utilizzano sen e cos perchè permettono al modello 
        di aiutarlo a capire la distanza tra due parole essendo parole che si trovano vicino 
        allora avranno risutlati simili rispetto a parole che si trovano nell aposizione 2 e 50  
        '''
        tokenVectiors = np.zeros((self.maxToken, self.d_model))
        
        for pos in range(len(self.input)):
            for i in range(0, self.d_model, 2):

                sin = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                tokenVectiors[pos, i] = sin

                # controlla solo se d_model è dispari non fa il cos dell'ultimo elemento del vettore
                if i + 1 < self.d_model: 
                    cos = math.cos(pos / (10000 ** (2 * i / self.d_model)))
                    tokenVectiors[pos, i + 1] = cos
        
        return tokenVectiors[:len(self.input)]
    
    def createTokenEmbedding(self):
        
        '''
        createTokenEmbedding ottiene i token usati come input nello spazio 
        vettoriale di self.tokenEmbedding
        '''
        embedding = np.zeros((len(self.input), self.d_model))

        for idToken, token  in enumerate(self.input):
            embedding[idToken] = self.tokenEmbedding[token] * math.sqrt(self.d_model)

        return embedding
    
    def getCombinedEmbedding(self):

        return self.positionalEmbedding + self.createTokenEmbedding()



if __name__ == '__main__':
    embedding = Embedding(512, [10, 30])
    # print(embedding.createPositionalEncoding())
    # print(embedding.createTokenEmbedding())

    print(embedding.getCombinedEmbedding())




