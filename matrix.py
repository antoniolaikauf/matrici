A = [[1,2,3],[4,5,6]]
B = [[7,8,9],[10,11,12]]
C = []

def add(a ,b, c):
    if(len(a) != len(b)): raise ValueError('dimensioni sbagliate')
    for x in range(len(a)):
        if(len(a[x]) != len(b[x])): raise ValueError('dimensioni sbagliate')
        row = [ a[x][y] + b[x][y] for y in range(len(a[x]))]
        c.append(row)
    return c
        
# print(add(A, B, C))

c = 3
def moltiplicazioneScalare(a, c):
    for x in a:
        x[:] = list(map(lambda n : n * c, x))
    return a 

# print(moltiplicazioneScalare(A, c))

def sottrazione(a, b, c):
    moltiplicazione = moltiplicazioneScalare(a, -1)
    somma = add(moltiplicazione, b, c)
    return somma 

print(sottrazione(A, B, C))