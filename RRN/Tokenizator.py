class Tokenizer:
    def __init__(self, word, numMerge = 50):
        self.word = word
        self.numMerge = numMerge
        self.idMerge = 256
        self.vocab = {x: x  for x in range(self.idMerge)}

    def countMaxPair(self, wordEncode):
        count = {}
        for letterId in range(len(wordEncode) - 1):
            pair = wordEncode[letterId], wordEncode[letterId + 1]
            if  pair not in count: count[pair] = 0
            else: count[pair] += 1
        return count
    
    def encode(self):
        wordEncode = list(self.word.encode('utf-8'))
        count = self.countMaxPair(wordEncode)
        maxPair = max(count, key=count.get)
        print(maxPair)
        
        while count[maxPair] > 1:
            print('cicici')
            
            wordId = 0
            while wordId < (len(wordEncode) - 1):
                if wordEncode[wordId] == maxPair[0] and wordEncode[wordId + 1] == maxPair[1]:
                    if maxPair not in self.vocab:
                        self.vocab[maxPair] = self.idMerge
                        self.idMerge += 1

                    wordEncode[wordId] = self.idMerge
                    del wordEncode[wordId + 1]  
                else:
                    wordId += 1

            print(wordEncode)               
            break
            
            
        print(count)
        
        pass
        
    def decode(self):
        pass

word = 'cicici'
tk = Tokenizer(word)
tk.encode()
print(tk.vocab)