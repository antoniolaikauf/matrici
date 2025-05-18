import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, tokensEmbedding, head = 8):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.d_model = d_model
        self.tokensEmbedding = tokensEmbedding
        self.dk = d_model // head
        self.dv = d_model // head
        self.w_q = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])
        self.w_k = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])
        self.w_v = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])

    def attention(self, q, k, v):

        '''
        attention è il blocco che permette di catturare relazioni tra i token di input 
        indipendementemente dalla loro posizione della frase, i token vengono processati 
        tutti insieme a differenza degli RRN che vengono processati ricursivamente.
        per ogni parola confronta la sua query con le key di tutte le altre parole,
        se la query di gatto è molto simile alla key di mangia allora mangia avrà un peso alto
        vedere immagine attention.png in image (ps nell'immagine manca la trasposizione di k in 
        realtà k sarebbe un arrey di 2x5 in cui parola esempio il sareebe 0.0 e 0.1 ma in colonna
        quindi uno sotto l'altro), ovviamente più l'output della moltiplicazione tra i due vettori
        è alto e più allora ci sarà una relazione tra di loro.
        dopo si fa la frazzione con self.dv per non permettere ai numeri di diventare troppo 
        grandi.
        la funzione softmax normalizza i pesi all'interno dei vettori in modo tale che la somma
        di ogni singolo vettore sia 1
        Supponiamo che per la parola gatto dopo lo scaedDotProduct sia 
        [0.01,0.07,0.092,0.02,0.06] (per"Il", "gatto", "mangia", "il", "pesce") e dopo la
        funzione sotmax l'output sia [0.15,0.20,0.25,0.16,0.19] si può notare che con la parola gatto il
        peso più grande sia mangia
        il risultato della softmax viene moltiplicato per v e si ottengono una nuova rappresentazione per ogni 
        token con stesse dimensioni di quelle iniziali  
        '''

        scaledDotPorduct = torch.matmul(q, torch.transpose(k, 0, 1)) / math.sqrt(self.dv)
        softmax = torch.softmax(scaledDotPorduct, dim=1)
        return torch.matmul(softmax, v)

    def forward(self):

        '''
        Ogni testa ha la sua matrice e quindi ha speecifiche q_w, k_w, v_w
        il vettore delle q (query) sarebbe cosa sto cercando (es. "gatto" chiede chi è collegato a lui).
        il vettore delle k (key) sarebbe chi sono io (es 'gatto' è un soggetto)
        il vettore delle v (value) sarebbe cosa porto (es 'gatto' è al centro della frase essendo che è il soggetto)
        '''

        output = []

        for headId in  range(self.head):
            q_w =  torch.matmul(self.tokensEmbedding, self.w_q[headId])
            k_w = torch.matmul(self.tokensEmbedding, self.w_k[headId])
            v_k = torch.matmul(self.tokensEmbedding, self.w_v[headId])
            head = self.attention(q_w, k_w, v_k)
            print(f'head number {headId + 1} shape {head.size()}')
            output.append(head)
        
        return torch.cat(output, dim=1)

class FFN:
    def __init__(self):
        pass

class Norm:
    def __init__(self):
        pass

if __name__ == '__main__':
    test = torch.tensor(torch.ones((2, 512)))
    heads = MultiHeadAttention(512, test)
    heads.forward()
