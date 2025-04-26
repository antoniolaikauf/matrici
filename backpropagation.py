x = 3.0
h = 0.001

def f(x):
    return x*2**3 - x + 4

# cambiamento nel valore della funzione quando ( x ) aumenta di un piccolo ( h )
change = f(x + h) - f(x)

# questo ci dice la pendenza della funzione in quel punto della derivata 
derivataSlope = (f(x + h) - f(x)) / h 



h1 = 0.0001

a = 3
b = 4
c = -3

d1 = a * b + c

a += h1 # DERIVATA RISPETTO AD A 
# b += h1 # DERIVATA RISPETTO AD B
# c += h1 # DERIVATA RISPETTO AD C

d2 = a * b + c

derivata = (d2 - d1) / h1 
# print(f"d1 = {d1}")
# print(f"d2 = {d2}")
# print(f"derivata {derivata}")

import math

class Value:
    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self._precedente = set(_children)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda: None
        self.name = ''

    # non ritorna un puntatore ma il suo valore 
    def __repr__(self):
        return f"value = {self.data}"
    
    def __rmul__(self, other):
        return self * other
    
    # a.__add__(b)
    def __add__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        '''
        video 1h e 10 min di backpropagation di andrej

        si calcola la chain rule 
        e quindi si calcola la derivata di o rispetto al nodo 
        se noi avessimo c = a + b e d = tanh(c) e volessimo calcolare la derivata di d rispetto ad a
        bisogna utilizare la chain rule e quindi prima si calcola la derivata di dd/da = dd/dc * dc/da 
        il gradiante di c rispetto ad a è sempre 1 (perche ((a + h) + b) - (a + b) / h ritorna h/h che fa 1)
        e quindi se il gradiante di dd/da è sempre dd/dc 
        '''
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward

        return out
    
    # a.__mul__(b)
    def __mul__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        '''
        video 1h e 10 min di backpropagation di andrej

        si calcola la chain rule 
        e quindi si calcola la derivata di o rispetto al nodo 
        se noi avessimo c = a * b e d = tanh(c) e volessimo calcolare la derivata di d rispetto ad a
        bisogna utilizare la chain rule e quindi prima si calcola la derivata di dd/da = dd/dc * dc/da 
        quindi la derivata di dd/dc sarebbe out.grad e invece se si calcola la derivata di a rispetto a c
        il risultato è b e quindi self.other
        '''
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __pow__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.data**other.data, (self,), '**')

        def _backward():
            self.grad += other.data * (self.data**(other.data - 1)) * out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += 1 * out.grad # gradiente della di se stesso è 1 
            other.grad += -1 * out.grad # il gradiente di other è -1 e non 1 provare a fare ((a - b + h) - (a -b)) / h il risultato è -1

        out._backward = _backward 

        return out

    
    # optimazier function 
    def tanh(self):
        # prende direttamente il valore senza che si passi come parametro 
        z = self.data
        t = (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out
        
    def backward(self):
        topo = []
        topoAlgorithm = set()

        def buildTopo(node):
            if node not in topoAlgorithm:
                topoAlgorithm.add(node)   
                for _prev in node._precedente:
                    buildTopo(_prev)
                topo.append(node)  
        
        self.grad = 1
        buildTopo(self)
        # reverse perchè si dovrebbe partire dall'output e quindi o
        for node in reversed(topo): 
            node._backward()
            # print(f'gradiante:  {node.grad}')
        
        
    
def lol():
    h1 = 0.001
    
    a = Value(4)
    b = Value(5)
    c = Value(-10)
    f = Value(-2)
    d = a * b
    e = d + c
    g = f * e
    print(f"valore di g senza modifiche {g}")

    a = Value(4 + h1)
    b = Value(5)
    c = Value(-10)
    f = Value(-2)
    d = a * b   
    e1 = d + c 
    g1 = f * e1
    print(f"valore di g1 cambiando a {g1}")

    derivata = (g1.data - g.data) / h1
    print(f"derivata {derivata}")

    # CALCOLO DELLE DERIVTE   
    g1.grad = 1
    f.grad = 10
    e1.grad = -2
    d.grad = -2
    c.grad = -2
    a.grad = -10
    b.grad = -8

    # PROCEDIMENTO DEL FOWARD PASS QUESTO SAREBBE UN STEP PER L'OTTIMIZZAZIONE 
    a.data += h1 * a.grad
    b.data += h1 * b.grad
    c.data += h1 * c.grad
    d = a * b   
    e2 = d + c 
    g2 = f * e2

    print(f"valore di g2 cambiando  a, b, c {g2}")

    '''
      per calcolare la derivata di e rispetto a c quindi dd/dc 
      il base case sarebbe cambiare d1 di h e ritornerebbe sempre 1 essendo che la
      sotrazzione tra d1 - d sarebbe h e h / h da 1 
    '''
    

def neuron():
    # input
    x1 = Value(2)
    x2 = Value(0)

    # weight
    w1 = Value(-3)
    w2 = Value(1)

    b = Value(6.8813735)

    x1w1 = x1 * w1; x1w1._op = '*'
    x2w2 = x2 * w2; x2w2._op = '*'

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._op = '+'
    n =  x1w1x2w2 + b
    o = n.tanh()

    print(o)
    # do / do
    o.grad = 1 
    # do / dn = 1 - tanh(n)**2
    n.grad = 1 - o.data**2 

    #  con addizioni il gradiante iniziale è sempre quelo precedente e quindi è uguale a quello di n 
    x1w1x2w2.grad = 0.5
    b.grad = 0.5

    x1w1.grad = 0.5
    x2w2.grad = 0.5

    # chain rule
    # se si calcolasse la derivata di x1w1 rispetto a w1 darebbe come risultato x1 
    w1.grad = x1.data * x1w1.grad
    x1.grad = w1.data * x1w1.grad
    w2.grad = x2.data * x2w2.grad 
    x2.grad = w2.grad * x2w2.grad

def backwardPropagation():
    '''
    per evitare un bug in cui se si usasse una variabile più di una volta il suo gradiante verrebbe sovrascritto 
    grazie ad una regola della chain rule questi gradianti si devono accumulare 
    (vedi video andrej 1:22)

    a = Value(4)
    b = Value(5)
    c = a + b 
    d = a * b 
    in questo caso se i gradianti non si accumulassero rimarrebbero queli per la moltiplicazione e quindi il gradiante di a 
    sarebbe sarebbe b (5) e quello di b sarebbe il valore di a (4) invece se si accumulano i gradianti sarebbero 6 per a e 5 per b 
    '''
    x1 = Value(2); x1.name = ' x1' 
    x2 = Value(0); x2.name = ' x2'

    # x2 = x1**x2

    # weight
    w1 = Value(-3); w1.name = ' w1'
    w2 = Value(1); w2.name = ' w2'

    b = Value(6.8813735); b.name = ' b'

    x1w1 = x1 * w1; x1w1._op = '*'; x1w1.name = ' x1w1'
    x2w2 = x2 * w2; x2w2._op = '*'; x2w2.name = ' x2w2'

    x1w1x2w2 = x2w2 + x1w1 ; x1w1x2w2._op = '+'; x1w1x2w2.name = 'x1w1x2w2'
    n =  x1w1x2w2 + b; n.name = 'n'
    o = n.tanh(); o.name = 'o'

    # bisogna inizializzarlo ad 1 essendo che è 0 
    o.grad = 1.0
    # o.backward()
    # n.backward()
    # x1w1x2w2.backward()
    # x1w1.backward()
    # x2w2.backward()
    # o.backward()

    o.backward()

# lol()
# neuron()

'''
per confermare che i nostri gradianbtiu sono corretti
'''

def pytorch():

    # import torch

    x1 = torch.Tensor([2.0]).double()      ; x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()      ; x2.requires_grad = True

    w1 = torch.Tensor([-3.0]).double()     ; w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()      ; w2.requires_grad = True
    b = torch.tensor([6.8813735]).double()   ; b.requires_grad = True
    n = (x1 * w1) + (x2 * w2) + b
    o = torch.tanh(n)

    print(f"o {o.data.item()}")

    o.backward()

    print(f"x1 {x1.grad.item()}")
    print(f"x2 {x2.grad.item()}")
    print(f"w1 {w1.grad.item()}")
    print(f"w2 {w2.grad.item()}")


backwardPropagation()

# pytorch()

import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # creazione delle weight in base a quanti nin (input) si ha 
        self.b = Value(random.uniform(-1, 1)) # creazione delle bias (solo una essendo che un neurona ha una sola bias) 

    def __call__(self, x): # creazione dell'output di ongi singolo neurone 
       act = sum((wn*xn for xn, wn in zip(x, self.w)), self.b)  # si fa la somma dell'array e si somma con self.b
       out = act.tanh()
       return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout): # nin sarebbe quanti input deve avere ogni singolo neurone dell'ouptut nout sarebbe quanti neuroni ha un singolo layer
        self.neurons = [Neuron(nin) for _ in range(nout)] # array del layer in base a quanti neuroni deve avere un layer

    def __call__(self, x):
        out = [n(x) for n in self.neurons] # calcolazione dell'output dei neuorni del layer 
        return out[0] if len(out) == 1 else out  
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            out = neuron.parameters()
            params.extend(out)

        return params 

class MLP: #Multi-Layer Perceptron
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # siuze rete neurale
        self.net = [Layer(sz[idxLayer], sz[idxLayer + 1]) for idxLayer in range(len(nouts))] # layout della rete neurale

    def __call__(self, x):
        for layer in self.net:
            x = layer(x) 
        
        return x
    
    def parameters(self):
        params = []
        for layer in self.net:
            out = layer.parameters()
            params.extend(out)
        return params

x = [2.0, 3.0, -1.0]
mlp = MLP(3, [4, 4, 1]) #la rete qua sarebbe 3 -> 4 -> 4 -> 1
# print(mlp(x)) # python farebbe n.__call__(x)

def training():
    x = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
         ]
    
    y = [1.0, -1.0, -1.0, 1.0]
    lr = 0.01 # learning rate

    # out = [mlp(idx) for idx in x]
    parametes = mlp.parameters()

    for _ in range(10):
        # foward 
        out = [mlp(idx) for idx in x]
        loss = sum(((Value(ygt) - yout)**2 for ygt, yout in zip(y, out)), Value(0))

        # backward
        for p in parametes:
            p.grad = 0.0

        loss.backward()

        # update
        for p in parametes:
           p.data -= lr * p.grad

        print(f'loss function: {loss}   output reteneurale = {out}')
    
    return out

print(training())