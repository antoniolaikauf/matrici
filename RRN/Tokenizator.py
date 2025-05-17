class Tokenizer:
    def __init__(self, word, numMerge = 50):
        self.word = word
        self.numMerge = numMerge
        self.idMerge = 255
        self.vocab = {x: x for x in range(256)}
        self.mergeDone = 0
        self.wordEncoded = 0

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
                    if maxPair not in self.vocab.values():
                        self.idMerge += 1
                        self.mergeDone += 1
                        self.vocab[self.idMerge] = maxPair

                    wordEncode[wordId] = self.idMerge
                    del wordEncode[wordId + 1]  
                else:  # si potrebbe togliere e inclementare l'indice sempre 
                    wordId += 1

            count = self.countPair(wordEncode)
            maxPair = max(count, key = count.get)

        self.wordEncoded = wordEncode
        print(f'new phrase: {self.wordEncoded} token mint: {self.mergeDone}')

        return self.wordEncoded
    
    def decode(self, wordDecoded):

        '''
        il decode prende come input un array di token e li trasforma
        il lettere, se si da come input un token che non è dentro al vocab 
        ritornerà un errore
        il ciclo finisce quando i token sono tutti minori di 256 essendo che se 
        sono maggiori allora c'è stato un merge e quando si incontra un token che 
        non è maggiore di 255 (e quindi un merge) si incrementa l'ìndice
        '''

        wordId = 0

        while wordId < len(wordDecoded):
            if wordDecoded[wordId] not in self.vocab: raise Exception(f' il token {wordDecoded[wordId]} non è presente in vocab')
            else:
                if (wordDecoded[wordId] > 255):
                    element = self.vocab[wordDecoded[wordId]]
                    wordDecoded[wordId] = element[0]
                    wordDecoded.insert(wordId + 1, element[1])
                else:
                    wordId += 1

        return ''.join(chr(token) for token in wordDecoded)

if __name__ == '__main__':
    word = 'cicici'
    tk = Tokenizer(word)
    token = tk.encode()
    phrase = [10, 256, 78]
    print(tk.decode(token))
    print(tk.decode(phrase))