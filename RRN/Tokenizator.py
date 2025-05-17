class Tokenizer:
    def __init__(self, word, numMerge = 50):
        self.word = word
        self.numMerge = numMerge
        self.idMerge = 255
        self.vocab = {x: x  for x in range(256)}
        self.mergeDone = 0

    def countPair(self, wordEncode):

        '''
        countPair funzione per calcolare la coppia massima 
        della frase prende le lettere a coppie di due e controlla
        se sono gia presenti all'interno del dictionary count, 
        se non ci sono le inserisce all'interno se ci sono incrementa il loro valore di 1
        '''

        count = {}
        for letterId in range(len(wordEncode) - 1):
            pair = wordEncode[letterId], wordEncode[letterId + 1]
            if  pair not in count: count[pair] = 1
            else: count[pair] += 1
        return count
    
    def encode(self):

        '''
        encode decodifica la frase in ascii quindi fino ad un massimo di 8
        bit (1 byte) da maxPair si ottiene la coppia massima e si fa il ciclo 
        finoa quando non ci sono più coppie che si ripetono nella frase.
        nella frase si sostituisce la coppia con il token minato e si 
        calcola le coppie con il nuovo token minato.
        wordId si incrementa solo se si trova un merge perchè quando noi 
        cancelliamo un token la frase si accorcia e anche se non ci 
        sarà mai una coppia con un nuovo token minato e la lettera successiva
        la controlliamo lo stesso cosi, volendo si potrebbe togliere l'else 
        e incrementare il id lo stesso che tanto non cambia 
        '''

        wordEncode = list(self.word.encode('utf-8'))
        count = self.countPair(wordEncode)
        maxPair = max(count, key = count.get)
        
        while count[maxPair] > 1 and self.mergeDone < self.numMerge:
            
            wordId = 0
            while wordId < (len(wordEncode) - 1):
                if wordEncode[wordId] == maxPair[0] and wordEncode[wordId + 1] == maxPair[1]:
                    if maxPair not in self.vocab:
                        self.idMerge += 1
                        self.mergeDone += 1
                        self.vocab[maxPair] = self.idMerge

                    wordEncode[wordId] = self.idMerge
                    del wordEncode[wordId + 1]  
                else:  # si potrebbe togliere e inclementare l'indice sempre 
                    wordId += 1

            count = self.countPair(wordEncode)
            maxPair = max(count, key = count.get)

        print(f'new phrase: {wordEncode} token mint: {self.mergeDone}')
        
    def decode(self):
        pass

word = 'cicici'
tk = Tokenizer(word)
tk.encode()
print(tk.vocab)