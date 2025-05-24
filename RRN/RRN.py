from Tokenizator import Tokenizer
from Embedding import Embedding
from Layer import MultiHeadAttention, FFN, add_Norm
import json
from datasets import load_dataset
my_dataset_dictionary = load_dataset("mideind/icelandic-english-translation")

class Encoder:
    def __init__(self, input, d_model = 512, N = 6):
        self.d_model = d_model
        self.N = N
        self.x = self.embedding(input)

    def embedding(self, x):
        
        embedding = Embedding(self.d_model, wordEncoded)
        wordsEmbedded = embedding.getCombinedEmbedding()
        # print(tokens.vocab)
        # print(tokens.decode(wordEncoded))
        return wordsEmbedded


    def Foward(self):

        '''
        nel encoder eseguito solo la foward essendo che se si usasse un architettura come 
        il trasformers non si è bisogno del backward, i parametri sono stati scelti in base 
        al paper attention is all you need e dal blocco dell'encoder si otterrà le h stato nascosto
        del paper Neural Machine Translation by Jointly Learning to Align and Translate
        '''

        inputs = self.x

        for layerId in range(self.N):
           attention = MultiHeadAttention(self.d_model, inputs)
           fowardAttention = attention.forward()

           addNorm1 = add_Norm(fowardAttention, inputs, self.d_model)
           residualConnection1 = addNorm1.residualConnection()
           norm1 = addNorm1.norm(residualConnection1)

           ffn = FFN(self.d_model, self.N, norm1)
           feedFoward = ffn.feedFoward(layerId)

           addNorm2 = add_Norm(feedFoward, norm1, self.d_model)
           residualConnection2 = addNorm2.residualConnection()
           norm2 = addNorm2.norm(residualConnection2)

           print(f'layer num {layerId}')
           inputs = norm2
        
        return inputs
        
    '''
    con l'architettura transformers non c'è bisogno della backward essendo che 
    ogni parola conosce la sua relazione con tutte le altre parole e 
    non solo quelle a destra e sinistra 
    '''
    # def backward(self):
        # pass

class  Decoder:
    def __init__(self):
        pass

    def foward(self):
        pass

class RRN:
    def __init__(self, x):
        self.x = x
        
    def encoder(self):
        h = Encoder(self.x)
        foward = h.Foward()
        # foward.backward()
        return foward

    def decoder(self):
        pass


if __name__ == '__main__':
    import os 
    x = my_dataset_dictionary['train'][10]['input']
    
    tokens = Tokenizer()
    phrases = {}
    for phraseId in range(len(my_dataset_dictionary['train'])):
        phrase = my_dataset_dictionary['train'][phraseId]
        wordEncoded = tokens.encode(phrase['input'])
        phrases[phraseId] = {
            'input': phrase['input'],
            'wordEncoded': wordEncoded,
            'target': phrase['target']
        }
    
    if os.path.exists('data.json') == False: 
        with open('data.json', 'w') as f:
            json.dump(phrases, f, ensure_ascii=False)
    
    with open('data.json', 'r') as f:
        print(json.load(f)['10'])

    
    # print(len(my_dataset_dictionary['train']))

    # rrn = RRN(x)
    # print(rrn.encoder())
