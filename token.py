# type: ignore
phrase = 'ciao, mi ciao'

class tokenization():
    
    def __init__(self, phrase):
        self.phrase = phrase

    def phraseAscii(self):
        return [ ord(c) for c in self.phrase]
    
    def merge(self):
        count = {}
        phrase = self.phraseAscii()
        for x in range(len(phrase)):
            pair = tuple(phrase[x : x + 2]) # trasformarlo in un tuple perchè non si possono usare direttamente le liste 
            count[pair] = count.get(pair, 0) + 1
        
        return count


token = tokenization(phrase)

print(token.merge())