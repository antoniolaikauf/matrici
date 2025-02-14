A = [[1,2,3],[4,5,6]]
B = [[7,8,9],[10,11,12]]
C = []

def add(a ,b, c):
    if(len(a) != len(b)): raise ValueError('dimensioni sbagliate')
    for x in range(len(a)):
        if(len(a[x]) != len(b[x])): raise ValueError('dimensioni sbagliate')
        d = [ a[x][y] + b[x][y] for y in range(len(a[x]))]
        c.append(d)
    return c
        

print(add(A, B, C))