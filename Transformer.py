import numpy as np
import math

class Tokenizer:
    def __init__(self):
        self.Token = 'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu'
        self.distribution = math.sqrt(6/256)
        self.maxToken = 256
        self.maxMerge = 10
    
    def tokenInput(self):
        return [ord(char) for char in self.Token]
    
    # create a dictionary to track all the pair and their appearances 
    def encode(self, tokens):
        dictionary = {}

        for idxToken in range(len(tokens) - 1):            
            pair = tuple(tokens[idxToken : idxToken + 2])
            if(pair not in dictionary ): dictionary[pair] = 1
            else: dictionary[pair] +=1

        return dictionary
    
    # merge the pair that appears more
    def merge(self):
        phrase = self.tokenInput()
        tokens = self.encode(phrase)
        maxPair = max(tokens, key=tokens.get)

        count = 0
        
        # cicle until you don't find any pair that appears more than two or ypu have finished the max number for merge 
        while (max(tokens.values()) > 1 and count < self.maxToken):
            print(f"Merging pair: {maxPair} with frequency: {tokens[maxPair]}")

            for idxPair in range(len(phrase) - 1):
                if(phrase[idxPair] == maxPair[0] and phrase[idxPair + 1] == maxPair[1]):
                    self.maxToken += 1
                    phrase[idxPair] = self.maxToken
                    del phrase[idxPair + 1]
                    break
            
            tokens = self.encode(phrase)
            count += 1
        
        print(f"The phrase is: {phrase}")
        return phrase

    def decode(self):
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
print(t.merge())
        
        