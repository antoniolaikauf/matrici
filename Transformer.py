import numpy as np
import math

class Tokenizer:
    def __init__(self):
        self.Token = 'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu'
        self.distribution = math.sqrt(6/256)
        self.maxToken = 256
        self.maxMerge = 10
        self.vocab =  {idx: tuple([idx]) for idx in range(256)}
    
    def tokenInput(self):
        return [ord(char) for char in self.Token]
    
    # create a dictionary to track all the pair and their appearances 
    def pairs(self, tokens):
        dictionary = {}

        for idxToken in range(len(tokens) - 1):            
            pair = tuple(tokens[idxToken : idxToken + 2])
            if (pair not in dictionary ): dictionary[pair] = 1
            else: dictionary[pair] +=1

        return dictionary
    
    # merge the pair that appears more
    def encode(self):
        phrase = self.tokenInput()
        tokens = self.pairs(phrase)
        maxPair = max(tokens, key=tokens.get)

        count = 0
        
        # cicle until you don't find any pair that appears more than two or ypu have finished the max number for merge 
        while ((max(tokens.values()) > 1) and (count < self.maxMerge)):
            print(f"Merging pair: {maxPair} with frequency: {tokens[maxPair]} count {count}")

            for idxPair in range(len(phrase) - 1):
                if(phrase[idxPair] == maxPair[0] and phrase[idxPair + 1] == maxPair[1]):
                    self.vocab[self.maxToken] = maxPair
                    self.maxToken += 1
                    phrase[idxPair] = self.maxToken
                    del phrase[idxPair + 1]
                    break
            
            tokens = self.pairs(phrase)
            count += 1
        
        print(f"The phrase is: {phrase}")
        return phrase

    def decode(self, phrase):
        count = 0
        while (self.maxMerge > count and (len(self.vocab) - 256) < count):

            count += 1
        pass
    
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
# print(t.encode())
print(t.encode())
# print(t.decode())
        
        