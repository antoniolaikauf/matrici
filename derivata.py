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


class Value:
    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self._precedente = set(_children)
        self.grad = 0.0

    # non ritorna un puntatore ma il suo valore 
    def __repr__(self):
        return f"value =  {self.data}"
    
    # a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # a.__mul__(b)
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    

def lol():
    h1 = 0.001
    
    a = Value(4)
    b = Value(5)
    c = Value(-10)
    d = a * b
    e = d + c

    a = Value(4 + h1)
    b = Value(5)
    c = Value(-10)

    d = a * b   
    e1 = d + c

    derivata = (e1.data - e.data) / h1

    print(f"d = {d}")
    print(f"d1 = {d1}")
    print(f"derivata {derivata}")

    '''
      per calcolare la derivata di e rispetto a c quindi dd/dc 
      il base case sarebbe cambiare d1 di h e ritornerebbe sempre 1 essendo che la
      sotrazzione tra d1 - d sarebbe h e h / h da 1 
    '''

lol()