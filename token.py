# type: ignore
phrase = 'ciao, mi ciao'

class Tokenizer():
    
    def __init__(self, phrase):
        self.phrase = phrase

    def phraseAscii(self):
        return [ ord(c) for c in self.phrase]
    
    def get_stats(self, phrase):
        count = {}
        for x in range(len(phrase)):
            pair = tuple(phrase[x : x + 2]) # trasformarlo in un tuple perchè non si possono usare direttamente le liste 
            count[pair] = count.get(pair, 0) + 1
        
        return count
    
    def merge(self):
        phrase = self.phraseAscii()
        stats = self.get_stats(phrase)
        maxPair = max(stats, key=stats.get)
        count = 256
        print(phrase)
        while stats and max(stats.values()) > 1:
            print("Merging pair:", maxPair, "with frequency:", stats[maxPair])
            stats.pop(maxPair)
            count += 1 
            x = 0
            while x < len(phrase) - 1:
                if phrase[x] == maxPair[0] and phrase[x + 1] == maxPair[1]:
                #  Sostituisci il primo elemento della coppia con il nuovo token...
                    phrase[x] = count
                    # ...e rimuovi il secondo elemento
                    del phrase[x + 1]
                    # Non aumentiamo x perché la lista si è accorciata
                else:
                    x += 1 
            #     if phrase[x:x + 2] == list(maxPair):
            #         phrase.pop(x + 1) 
            #         phrase[x] = count
            print(phrase)
            stats = self.get_stats(phrase)
            print(maxPair)
            maxPair = max(stats, key=stats.get)
            print(stats)


        


token = Tokenizer(phrase)

print(token.merge())
