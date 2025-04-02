import numpy as np
from numpy.random import default_rng
import math

class Tokenizer:
    def __init__(self):
        self.distribution = math.sqrt(6/256)
        self.maxToken = 256
        self.maxMerge = 10
        self.vocab =  {idx: tuple([idx]) for idx in range(256)}
    
    def tokenInput(self, tokens):
        return [ord(char) for char in tokens]
    
    # create a dictionary to track all the pair and their appearances 
    def pairs(self, tokens):
        dictionary = {}

        for idxToken in range(len(tokens) - 1):            
            pair = tuple(tokens[idxToken : idxToken + 2])
            #if the pair doesn't increse we create a new pair 
            if (pair not in dictionary ): dictionary[pair] = 1
            # if the pair exist we increase by 1
            else: dictionary[pair] +=1

        return dictionary
    
    # merge the pair that appears more
    def encode(self, phrase):
        tokens = self.tokenInput(phrase)
        tokensPairs = self.pairs(tokens) # all possible merge 
        maxPair = max(tokensPairs, key=tokensPairs.get) # you get the torque that appears the most

        count = 0
        
        # cicle until you don't find any pair that appears more than two or the possible merge have finisched 
        while ((max(tokensPairs.values()) > 1) and (count <= self.maxMerge)):

            idxPair = 0
            #cile on the phrase 
            while idxPair < len(tokens) - 1:
                print(f"Merging pair: {maxPair}")
                #if we have a match we change it with a merge
                if tokens[idxPair] == maxPair[0] and tokens[idxPair + 1] == maxPair[1]:

                    tokenToUse = None
                    #if we have a merge inside our vocab that is equal to maxPair we take that 
                    for token, pair in self.vocab.items():
                        if pair == maxPair:
                            tokenToUse = token  # Uso il token esistente
                            break

                    # if we don't have a merge inside our vocab we create a new token  
                    if tokenToUse == None:
                        tokenToUse = self.maxToken
                        self.vocab[self.maxToken] = maxPair
                        self.maxToken += 1

                    # change the pair with the token mint
                    tokens[idxPair] = tokenToUse
                    del tokens[idxPair + 1] 

                else:
                    # we increase the idxPair only if we don't find a match if we shrink the phrase we don't want to increase 
                    # the index because could skip some number  
                    idxPair += 1 
                    
            # we create the new pair token with the new merge 
            tokensPairs = self.pairs(tokens)
            maxPair = max(tokensPairs, key=tokensPairs.get)

            count += 1
        
        # print(f"The phrase is: {tokens}")
        return tokens

    def decode(self, phrase, text= False):
        # the idx we increase by 1 because if a pair es. 276 is the merge between 256 and 257 if we increase by two 
        # we could skip the element 257 that is a merge 

        idx = 0

        # while loop over all the element in phrase 
        while (idx < len(phrase)):
            
            # check if the element exist in the dictionary     
            assert self.vocab.get(phrase[idx]) != None , f'carattere non esiste nel dizionario {phrase[idx]}'
            # take the pair in the vocab
            element = self.vocab.get(phrase[idx])
            # we change the element in phrase 
            phrase[idx] = element[0]
            if len(element) > 1:
            # if the len of the pair is more than 1 we change the second element
                phrase.insert(idx + 1, element[1])


            # we do't increase th number of idx until is not a merge because if the phrase[idx] > 256 we know that is a merge and we stay here        
            if phrase[idx] < 255:
                idx += 1
            
        # convert to char
        if (text): phrase = ''.join(chr(x) for x in phrase)

        return phrase
    
    def mergeFile(self):
        print(self.vocab)
        with open('file.txt', 'w') as dati:
            for key, token in self.vocab.items():
                dati.write(f"the key {key} -> merge {token}\n")


class MultiHeadAttention:
    def __init__(self, tokensVector):
        # length of each token vector 
        self.d_model = 512
        self.tokensVector = tokensVector
        # number of head
        self.headNumber = self.d_model // 64
        self.d_k = self.d_model // self.headNumber
        #weith of every query, value , key for every head
        self.w_q = [default_rng(10 + x).random((self.d_model, self.d_k)) * 0.001 for x in range(self.headNumber)]
        self.w_k = [default_rng(20 + x).random((self.d_model, self.d_k)) * 0.001 for x in range(self.headNumber)]
        self.w_v = [default_rng(30 + x).random((self.d_model, self.d_k)) * 0.001 for x in range(self.headNumber)]

    def multiHead(self):
        heads = []
        for head in range(self.headNumber):
            # those are the key, value and query of every token but with dimension len of vectorToken x 512 
            wk = np.dot(self.tokensVector, self.w_k[head]) 
            wv = np.dot(self.tokensVector, self.w_v[head]) 
            wq = np.dot(self.tokensVector, self.w_q[head]) 
            h = self.head(wk, wv, wq) # return 64 token of every words so an vector of 6x64        
            heads.append(h)

        # we concatenate every chunk of 64 
        # forst word [[a1, a2, a3] and  [[b1, b2, b3]
        #second word [a4, a5, a6]]      [b4, b5, b6]]

        #now the array is 
        #[[a1, a2, a3, b1, b2, b3],
        #[a4, a5, a6, b4, b5, b6]]
        
        concat = np.concatenate(heads, axis=-1) 
        # print("Concatenated output shape:", concat)
        return concat  
        
    def softmax(self, x):
        # exp of every element
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        # sum of every exp element th exp is 2.7
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def head(self, k, v, q):
        mol = np.dot(q , np.transpose(k))
        div = mol / math.sqrt(self.d_k)
        # print(f"softmax: {self.softmax(div)}")
        Smax = np.dot(self.softmax(div), v)      
        # the value of Smax if sum are equal to 1, thanks to the softamx function   
        return Smax

#feed foward
class  FFN:
    def __init__(self, multiHead, d_model):
        # matrici di dimensioni 512x2048 per prima trasformazione lineare
        self.w = [default_rng(10 + x).random((d_model, d_model * 4)) * 0.001 for x in range(len(multiHead))]
        self.b= [default_rng(20 + x).random((1, d_model * 4)) * 0.001 for x in range(len(multiHead))]
        # matrici di dimensioni 2048x512 per seconda trasformazione lineare cosi da riportare l'output a 512
        self.b2= [default_rng(20 + x).random((1, d_model)) * 0.001 for x in range(len(multiHead))]
        self.w2 = [default_rng(10 + x).random((d_model * 4, d_model)) * 0.001 for x in range(len(multiHead))]

        self.multihead = multiHead
        
    def Relu(self, x):
        for index, value in enumerate(x[0]): 
            if value < 0: x[index] = 0
        return x

    def  linearTransNetwork(self):
        output = []
        for index, value in enumerate(self.multihead):
            # value*W + b
            mul = np.dot(value, self.w[index])
            add = np.add(mul, self.b[index])
            # risultato inserito nella funzione RELU 
            firstLinear = self.Relu(add)
            # value*W + b
            mul2 = np.dot(firstLinear, self.w2[index])
            add2 = np.add(mul2, self.b2[index])

            output.append(add2)

        return output


class Embedding:
    def __init__(self, tokens):
        self.d_model = 512
        self.tokens = tokens
        self.w = [default_rng(10 + idx).random((1, self.d_model)) * math.sqrt(self.d_model) for idx in range(len(self.tokens))] 

    # visto che nell'architettura del transformers il modello processa tutti
    # i token tutti insieme e non uno alla volta come i modelli
    # RRM deve sapere dove le parole si trovano nella frase  
    def positionEncodig(self):
        vectorTokens = []
        for token in range(len(self.tokens)):
            vector = [] # vettore per ogni token 
            for i in range(self.d_model // 2):
                # i non consiste nella posizione del vettore ma sarebbe il numero
                # della coppia e visto che il vettore è lungo d_model
                # allora si fa diviso due essendo che si inseriscono due evalori 
                valueSin = math.sin(token / 10000**(2 * i / self.d_model)) # questo corrisponderebbe per pari 
                vector.append(valueSin)

                valueCos = math.cos(token / 10000**(2 * i / self.d_model)) # questo corrisponderebbe per dispari 
                vector.append(valueCos)

            vectorTokens.append(vector)
        return vectorTokens
class Transformers:
    def __init__(self):
        self.d_model = 512
        self.tokenizer = Tokenizer()

    def encoder(self):
        tokens = self.tokenizer.encode('ciao')
        
        embedding = Embedding(tokens)
        
        multiHeadAttention = MultiHeadAttention(embedding.positionEncodig())
        ffn = FFN(multiHeadAttention.multiHead(), self.d_model)
        return ffn.linearTransNetwork()


t = Transformers()

# print(t.encoder())
t.encoder()

        
#The third is the path length between long-range dependencies in the network. Learning long-range
#dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
#ability to learn such dependencies is the length of the paths forward and backward signals have to
#traverse in the network. The shorter these paths between any combination of positions in the input
#and output sequences, the easier it is to learn long-range dependencies

# il paper sostiene che più la sequenza di parole è lunga e più il modello tende a sbagliare/dimenticare 
# e quindi con architetture come i trasnformer non c'è bisogno di saltare tra parole perchè processa l'intera 
# frase e quindi deve fare un solo balzo rispetto ad altre architetture 

# As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially
# executed operations, whereas a recurrent layer requires O(n) sequential operations

#To improve computational performance for tasks involving
#very long sequences, self-attention could be restricted to considering only a neighborhood of size r in
#the input sequence

# es. 
# "Il":     [x, x, 0, 0, 0]
# "cane":   [x, x, x, 0, 0]
# "corre":  [0, x, x, x, 0]
# "veloce": [0, 0, x, x, x]
# "ora":    [0, 0, 0, x, x]
# qua ogni token guarderebbe solo i token che ha in parte e quindi con r uguale a 2