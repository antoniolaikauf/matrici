import numpy as np
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
            if (pair not in dictionary ): dictionary[pair] = 1
            else: dictionary[pair] +=1

        return dictionary
    
    # merge the pair that appears more
    def encode(self, phrase):
        tokens = self.tokenInput(phrase)
        tokensPairs = self.pairs(tokens)
        maxPair = max(tokensPairs, key=tokensPairs.get)

        count = 0
        
        # cicle until you don't find any pair that appears more than two or ypu have finished the max number for merge 
        while ((max(tokensPairs.values()) > 1) and (count < self.maxMerge)):
            print(f"Merging pair: {maxPair} with frequency: {tokensPairs[maxPair]} count {count}")

            for idxPair in range(len(tokens) - 1):
                if(tokens[idxPair] == maxPair[0] and tokens[idxPair + 1] == maxPair[1]):
                    self.vocab[self.maxToken] = maxPair
                    tokens[idxPair] = self.maxToken
                    del tokens[idxPair + 1]
                    self.maxToken += 1
                    break
            
            tokensPairs = self.pairs(tokens)
            count += 1
        
        print(f"The phrase is: {tokens}")
        return tokens

    def decode(self, phrase, text= False):
        phraseDecode = phrase
        print(self.vocab)
        print(phrase)
        for x in range(len(phrase)):
            element = self.vocab.get(phrase[x])
            assert element != None, f"non esiste merge {phrase[x]}"
            phraseDecode[x] = element[0]
            if len(element) > 1: phraseDecode.insert(x + 1, element[1])
        
        if(text): phraseDecode = ''.join(chr(x) for x in phraseDecode)
        return phraseDecode
    
    def tokenInputVector(self):
        return {tk : np.full(256, self.distribution) for tk in self.tokenInput()}



class Transformers(Tokenizer):
    def __init__(self):
        super().__init__()

    def encoder(self):
        pass
        # print(self.distribution)
        # print(self.tokenInputVector())

t =Transformers()
print(t.decode(t.encode('cici'), True))
# print(t.decode(t.encode('cici'), True))
        
        