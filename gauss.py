matrix = [[1,3,5,6,7],[0,7,0,4,5],[1,4,5,0,6]]

def gauus(M):
    for x in range(len(M) - 1):
        # scambio della riga 
        if M[x][0] == 0: 
            i = 1
            while M[x + i][0] == 0 : i+=1
            value = M[x + i] 
            M[x + i] = M[x]
            M[x] = value
        
        for 

    return M


print(gauus(matrix))