import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head = 8):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.d_model = d_model
        self.dk = d_model // head
        self.dv = d_model // head
        self.w_q = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk), requires_grad=True ) for _ in range(head)])
        self.w_k = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk), requires_grad=True ) for _ in range(head)])
        self.w_v = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk), requires_grad=True ) for _ in range(head)])

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

        scaled_dot_product = torch.matmul(q, torch.transpose(k, 0, 1)) / math.sqrt(self.dv)
        softmax = torch.softmax(scaled_dot_product, dim=1)
        return torch.matmul(softmax, v)

    def forward(self, tokens_embedding):

        '''
        Ogni testa ha la sua matrice e quindi ha speecifiche q_w, k_w, v_w
        il vettore delle q (query) sarebbe cosa sto cercando (es. "gatto" chiede chi è collegato a lui).
        il vettore delle k (key) sarebbe chi sono io (es 'gatto' è un soggetto)
        il vettore delle v (value) sarebbe cosa porto (es 'gatto' è al centro della frase essendo che è il soggetto)
        '''

        output = []

        for headId in  range(self.head):
            q_w =  torch.matmul(tokens_embedding, self.w_q[headId])
            k_w = torch.matmul(tokens_embedding, self.w_k[headId])
            v_k = torch.matmul(tokens_embedding, self.w_v[headId])
            head = self.attention(q_w, k_w, v_k)
            print(f'head number {headId + 1} shape {head.size()}')
            output.append(head)
        
        return torch.cat(output, dim=1)

class FFN(nn.Module):
    def __init__(self, d_model, N):
        super(FFN, self).__init__()
        self.d_model_intermediate = 2048
        self.d_model = d_model
        # nel paper dicono che si utilizzano differenti parametri in base al layer, quindi ogni layer avra differenti arametri rispetto agli altri
        self.weight_1 = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.d_model_intermediate), requires_grad=True ) for _ in range(N)])
        self.weight_2 = nn.ParameterList([nn.Parameter(torch.randn(self.d_model_intermediate, self.d_model), requires_grad=True ) for _ in range(N)])
        self.bias_1 = nn.ParameterList([nn.Parameter(torch.randn(1, self.d_model_intermediate), requires_grad=True ) for _ in range(N)])
        self.bias_2 = nn.ParameterList([nn.Parameter(torch.randn(1, self.d_model), requires_grad=True ) for _ in range (N)])
    
    def feedFoward(self, index, input):

        '''
        ffn sarebbero due trasformazioni lineari la prima ha una dimensione di 2048 una volta inserita 
        all'interno della funzione relu che azzera tuttti i valori che sono minori di 0  la seconda trasformazione
        lineare ha dimensione 512
        la ffn viene eseguita per introdurre la non linearità, le proprietà della linearità sono:
        a = 3 b = 4 scalare = 3
        - f(a + b) = f(a) + f(b)
        - f(a * scalare) = scalare * f(a)
        '''
        
        linear_trasformation_1 = torch.add(torch.matmul(input, self.weight_1[index]), self.bias_1[index])
        softmax = torch.relu(linear_trasformation_1)
        linear_trasformation_2 = torch.add(torch.matmul(softmax, self.weight_2[index]), self.bias_2[index])
        return linear_trasformation_2  

class add_Norm(nn.Module):
    def __init__(self, d_model = 512):
        super(add_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def residualConnection(self, out_sub_layer, in_sub_layer):

        '''
        la residual connection viene usata per far si che i gradianti 
        non diventano valori troppo piccoli e quindi rallentare 
        l'ottimmizzazione, questo perchè i gradianti dei neuroni che si trovano 
        all'inizio dell'input vengono calcolati tramite la chain rule e quindi se i
        gradianti dei layer che sono lontani da quelli iniziali (quelli dell'output) sono 
        gi adi per se piccoli allora i gradianti dei layer che sono all'inizio dell'input 
        saranno bassi o quasi zero 
        '''

        return torch.add(in_sub_layer, out_sub_layer)

    def norm(self, residual_connection):

        '''
        Applica la LayerNorm al risultato della connessione residua per stabilizzare le attivazioni,
        evitando valori troppo grandi o instabili.
        layer_norm lavora sulla dimensione del d_model quindi 512
        quindi prima calcola la media, in seguito calcola la varianza (si calcola facendo il valore - media elevato alla seconda).
        Per ogni valore all'interno del vettore del token calcola (x - media / radiceQuadtrata(varianzacalcola + eps)) * gamma + beta
        gamma e beta sono parametri apprendibili invece eps sarebbe un valore che tende a 0 per evitare divisione per 0
        '''
     
        return self.layer_norm(residual_connection)
        


if __name__ == '__main__':
    d_model = 512
    test = torch.tensor(torch.ones((2, d_model)))
    heads = MultiHeadAttention(d_model)
    out = heads.forward(test)

    addNorm = add_Norm(d_model)
    residual_connection = addNorm.residualConnection(out, test)
    print(addNorm.norm(residual_connection))
    
    print(out.size())

    test1 = FFN(d_model, 6)
    ffn = test1.feedFoward(2, out)
    print(test1.weight_1[2])
    print(ffn.size())
