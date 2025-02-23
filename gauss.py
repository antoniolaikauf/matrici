# type: ignore
matrix1 = [[1,2,3,1,1,4],[1,2,3,2,3,-2],[1,1,1,1,1,-2],[-3,-5,-7,-4,-5,0]]
matrix = [[0,1,-1,0],[1,2,0,1],[2,-1,1,2]] 

def gauss(M):
    for x in range(len(M) - 1):
        # scambio della riga 
        if M[x][x] == 0 : 
            i = 1
            while M[x + i][x] == 0 : i+=1
            value = M[x + i] 
            M[x + i] = M[x]
            M[x] = value
        for y in range(x + 1, len(M)):
            if M[y][x] != 0:
                A = -1 * (M[y][x] / M[x][x])
                for o in range(len(M[y])):
                    M[y][o] =  M[y][o] + (A * M[x][o])

    return M


print(f'sistema finale {gauss(matrix)}')