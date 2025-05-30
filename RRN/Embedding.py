import math
import torch
import torch.nn as nn

'''
d_model sarebbe la grandezza dei vettori con cui si lavora all'interno del modello 

vocab_size sarebbe la quantità di tutti i token possibili del modello 

max_token sarebbe la grandezza di quanti token può gestire in una sola 
volta il modello 

token_embedding sarebbe lo spazio vettoriale in cui i token (max_token) sono 
rappresentati, qua parole con significato simile saranno molto vicine 
rispetto a parole che hanno sigificato completamente diverso

es.
cane avrà un vettore molto simile a quello di gatto  rispetto al vettore
della parola tavolo e quindi nello spazio vettoriale la parola cane sarà  
vicino alla parola gatto 
'''

class Embedding:
    def __init__(self, d_model, input, vocab_size = 10000, max_token = 512):
        self.d_model = d_model
        self.input = input
        self.vocab_size = vocab_size
        self.max_token = max_token
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = self.createPositionalEncoding()

    def createPositionalEncoding(self):

        '''
        createPositionalEncoding permette di creare una matrice di vettori 
        delle posizioni dei token all'interno della frase. Crea una matrice di 
        dimensione (Max_token, d_model) ma restituisce solo una parte di essa che sarebbe
        quella con solo le posizioni dei token 
        nel paper attention is all you need utilizzano sen e cos perchè permettono al modello 
        di aiutarlo a capire la distanza tra due parole essendo parole che si trovano vicino 
        allora avranno risutlati simili rispetto a parole che si trovano nell aposizione 2 e 50  
        '''
        
        token_vectiors = torch.zeros((self.max_token, self.d_model))
        
        for pos in range(len(self.input)):
            for i in range(0, self.d_model, 2):

                sin = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                token_vectiors[pos, i] = sin

                # controlla solo se d_model è dispari non fa il cos dell'ultimo elemento del vettore
                if i + 1 < self.d_model: 
                    cos = math.cos(pos / (10000 ** (2 * i / self.d_model)))
                    token_vectiors[pos, i + 1] = cos
        
        return token_vectiors[:len(self.input)]
    
    def createTokenEmbedding(self):
        
        '''
        createTokenEmbedding ottiene i token usati come input nello spazio 
        vettoriale di self.tokenEmbedding come input si usa un tesor con 
        i token 
        '''

        return self.token_embedding(torch.tensor(self.input)) * math.sqrt(self.d_model)
    
    def getCombinedEmbedding(self):

        '''
        combinazione tra la codifica della frase e l'embedding dei token 
        questo permette al modello di apprendere la posizione essenodo che due token vicini
        avranno risultati simili nella codifica delle posizionbe rispetto a un token 
        in posizione 2 e uno a 50.
        Permette di capire anche il significato essendo che nel tokenEmbedding i token 
        con significato simile saranno vicino di posizione         
        '''

        return torch.add(self.positional_embedding, self.createTokenEmbedding())



if __name__ == '__main__':
    embedding = Embedding(512, [10, 30])
    # print(embedding.createPositionalEncoding())
    # print(embedding.createTokenEmbedding())
    # print(embedding.tokenEmbedding)

    print(embedding.getCombinedEmbedding())




