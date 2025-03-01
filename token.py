# type: ignore
phrase = 'ciao ciaoo ci'

class Tokenizer():
    
    def __init__(self, phrase):
        self.phrase = phrase
        self.vocab = {}

    def phraseAscii(self):
        return list(phrase.encode('utf-8'))
    
    def get_stats(self, phrase):
        count = {}
        for x in range(len(phrase)):
            pair = tuple(phrase[x : x + 2]) # trasformarlo in un tuple perchè non si possono usare direttamente le liste 
            count[pair] = count.get(pair, 0) + 1
        
        return count
    
    def encode(self):
        phrase = self.phraseAscii() 
        # si prendono le coppie maggiori 
        stats = self.get_stats(phrase) 
        maxPair = max(stats, key=stats.get)
        # massimo che può raggiungere ASCII
        count = 256 
        # stats maggiori di 1 per fare il merge 
        while max(stats.values()) > 1:
            print(f"Merging pair: {maxPair} with frequency: {stats[maxPair]}")
            count += 1 
            self.vocab[count] = maxPair

            x = 0
            while x < len(phrase) - 1:
                if phrase[x] == maxPair[0] and phrase[x + 1] == maxPair[1]:
                    # sostituzione della prima coppia dalla frase 
                    phrase[x] = count
                    # rimozione della seconda coppia dalla frase
                    del phrase[x + 1] 
                else:
                    x += 1 
            # calcolo di nuovo le coppie dopo aver effettuato il merge 
            stats = self.get_stats(phrase)
            # presa della nuova coppia 
            maxPair = max(stats, key=stats.get)
        
        print(f"Merge effettuati : {count - 256}")
        return phrase
    
    def decode(self):
        merge = self.vocab # merge salvati
        phrase = self.encode()
        count = 0

        while(count != len(merge)):
            count += 1
            i = 0
            while i != len(phrase):
                # si trova che l'elemento della frase sia un merge e quindi maggiore di 256
                if phrase[i] > 256:
                    element = merge.get(phrase[i])
                    # sostituzione
                    phrase[i] = element[0]
                    phrase.insert(i + 1, element[1])
                i+= 1
            print(phrase)
        
        print(self.phraseAscii() == phrase)
        return bytes(phrase)


token = Tokenizer(phrase)

print(token.encode())
print(token.decode())
