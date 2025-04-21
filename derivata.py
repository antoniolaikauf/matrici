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
        self.backward = lambda: None

    # non ritorna un puntatore ma il suo valore 
    def __repr__(self):
        return f"value = {self.data}"
    
    # a.__add__(b)
    def __add__(self, other):
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
            self.grad = 1 * out.grad
            self.other = 1 * out.grad

        out.backward = _backward

        return out
    
    # a.__mul__(b)
    def __mul__(self, other):
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
            self.grad = other.data * out.grad
            self.other = self.data * out.grad

        out.backward = _backward

        return out
    
    # optimazier function 
    def tanh(self):
        # prende direttamente il valore senza che si passi come parametro 
        z = self.data
        t = (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - out.data**2) * out.grad

        out.backward = _backward

        return out
        
        
        

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

    # PROCEDIMENTO DEL FOWARD PASS QUESTO SAREBBE UN STEPPER L'OTTIMIZZAZIONE 
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

    # bisogna inizializzarlo ad 1 essendo che è 0 
    o.grad = 1.0
    o.backward()
    # o.backward()
    print(f'gradiante di n {n.grad}')

lol()
# neuron()
backwardPropagation()