from Tokenizator import Tokenizer
from Embedding import Embedding
from Layer import MultiHeadAttention, FFN, add_Norm
import json
import os 
from datasets import load_dataset
my_dataset_dictionary = load_dataset("mideind/icelandic-english-translation")

class encoderLayer:

    '''
    encoderLayer sarebbe la struttura di un singolo layer composto da 
    attention, ffn, e due add_norm come descritto dal paper.
    la funzione foward consiste nel processo di elaborazione dei dati 
    di un singolo layer 
    '''

    def __init__(self, d_model, heads=8, N=6):
        self.attention = MultiHeadAttention(d_model, heads)
        self.ffn = FFN(d_model, N)
        self.add_norm1 = add_Norm(d_model)
        self.add_norm2 = add_Norm(d_model)
        self.N = N

    def foward(self, x, indexHead):
        
        attention_out = self.attention.forward(x)
        residual_connection1 = self.add_norm1.residualConnection(attention_out, x)
        norm1_out = self.add_norm1.norm(residual_connection1)

        ffn_out = self.ffn.feedFoward(indexHead, norm1_out)

        residual_connection2 = self.add_norm2.residualConnection(ffn_out, norm1_out)
        norm2_out = self.add_norm2.norm(residual_connection2)

        return norm2_out

class Encoder:
    def __init__(self, vocab_size, d_model = 512, N = 6):
        self.d_model = d_model
        self.N = N
        self.vocab_size = vocab_size
        self.net_encoder = [encoderLayer(self.d_model, 8, self.N) for _ in range(N)] # composizione della rete per l'encoder

    def embedding(self, x):
        
        embedding = Embedding(self.d_model, x, self.vocab_size)
        words_embedded = embedding.getCombinedEmbedding()
        # print(tokens.vocab)
        # print(tokens.decode(wordEncoded))
        return words_embedded


    def Foward(self, x):

        '''
        nel encoder eseguito solo la foward essendo che se si usasse un architettura come 
        il trasformers non si è bisogno del backward, i parametri sono stati scelti in base 
        al paper attention is all you need e dal blocco dell'encoder si otterrà le h stato nascosto
        del paper Neural Machine Translation by Jointly Learning to Align and Translate
        '''

        inputs = self.embedding(x)

        for layer_id in range(len(self.net_encoder)):
           layer_out = self.net_encoder[layer_id].foward(inputs, layer_id)

           print(f'layer num {layer_id}')
           inputs = layer_out
        
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
    def __init__(self, vocab_size, d_model = 512, N = 6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.N = N
        
    def encoder(self, x):
        h = Encoder(self.vocab_size)
        foward = h.Foward(x)
        # foward.backward()
        return foward

    def decoder(self):
        pass


if __name__ == '__main__':
    
    tokens = Tokenizer()
    phrases = {}
    for phraseId in range(len(my_dataset_dictionary['train'])):
        phrase = my_dataset_dictionary['train'][phraseId]
        word_encoded = tokens.encode(phrase['input'])
        phrases[phraseId] = {
            'input': phrase['input'],
            'wordEncoded': word_encoded,
            'target': phrase['target']
        }
    
    if os.path.exists('data.json') == False: 
        with open('data.json', 'w') as f:
            json.dump(phrases, f, ensure_ascii=False)
    
    with open('data.json', 'r') as f:
        rrn = RRN(len(tokens.vocab))
        words = json.load(f)['10']['wordEncoded']
        out = rrn.encoder(words)
        print(out)

    # print(tokens.vocab)
    # print(len(my_dataset_dictionary['train']))
