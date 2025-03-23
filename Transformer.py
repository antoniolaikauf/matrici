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
        
        print(f"The phrase is: {tokens}")
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
    
    def tokenInputVector(self):
        return {tk : np.full(256, self.distribution) for tk in self.tokenInput()}
    
    def mergeFile(self):
        print(self.vocab)
        with open('file.txt', 'w') as dati:
            for key, token in self.vocab.items():
                dati.write(f"the key {key} -> merge {token}\n")


class Transformers(Tokenizer):
    def __init__(self):
        super().__init__()

    def encoder(self):
        pass
        # print(self.distribution)
        # print(self.tokenInputVector())

t =Transformers()
# t.encode('fffffffffffffffffffuuuuuuuuuuuuuuuu')
print(t.decode(t.encode('fffffffffffffffffffuuuuuuuuuuuuuuuu'), True))
t.mergeFile()
# print(t.decode([76, 500], True))

        
        