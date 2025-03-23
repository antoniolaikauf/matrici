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
            #if the pair doesn't increse we create a new pair 
            if (pair not in dictionary ): dictionary[pair] = 1
            # if the pair exist we increase by 1
            else: dictionary[pair] +=1

        return dictionary
    
    # merge the pair that appears more
    def encode(self, phrase):
        tokens = self.tokenInput(phrase)
        tokensPairs = self.pairs(tokens)
        maxPair = max(tokensPairs, key=tokensPairs.get)

        count = 0
        
        # cicle until you don't find any pair that appears more than two or ypu have finished the max number for merge 
        while ((max(tokensPairs.values()) > 1) and (count <= self.maxMerge)):
            print(f"Merging pair: {maxPair} with frequency: {tokensPairs[maxPair]}")

            for idxPair in range(len(tokens) - 1):
                # if we have a match we change the phrase and we put in the merge 
                if(tokens[idxPair] == maxPair[0] and tokens[idxPair + 1] == maxPair[1]):
                    self.vocab[self.maxToken] = maxPair
                    tokens[idxPair] = self.maxToken
                    # we delete the extra token 
                    del tokens[idxPair + 1]
                    self.maxToken += 1
                    break
            # we create the new pair token with the new merge 
            tokensPairs = self.pairs(tokens)
            maxPair = max(tokensPairs, key=tokensPairs.get)
            
            count += 1
        
        print(f"The phrase is: {tokens}")
        return tokens

    def decode(self, phrase, text= False):
        # the idx we increase by 1 because if a pair es. 276 is the merge between 256 and 257 if we increase by two 
        # we could skip the element 257 that is a merge 

        idx = 0
        # check if the element exist in the dictionary
        assert self.vocab.get(phrase[idx]) != None , f'carattere non esiste nel dizionario {phrase[idx]}'

        # while loop over all the element in phrase 
        while (self.vocab.get(phrase[idx]) and idx != len(phrase) - 1):
            # take the pair in the vocab
            element = self.vocab.get(phrase[idx])
            # we change the element in phrase 
            phrase[idx] = element[0]
            if len(element) > 1:
            # if the len of the pair is more than 1 we change the second element
                phrase.insert(idx + 1, element[1])
        
            idx += 1
            
            # check if the element exist in the dictionary
            assert self.vocab.get(phrase[idx]) != None , f'carattere non esiste nel dizionario {phrase[idx]}'
        # convert to char
        if (text): phrase = ''.join(chr(x) for x in phrase)

        return phrase
    
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
print(t.decode(t.encode('uuuuuuuuuuuuuuuuuuufffffffffffff'), True))
# print(t.decode([76,600], True))
# print(t.decode(t.encode('cici'), True))
        
        