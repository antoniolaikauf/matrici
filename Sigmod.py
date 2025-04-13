import math

#------------------------------------
# VALORE GRANDE NEGATIVO
#------------------------------------

w = -20
b = -30
x = 0.63
z = x*w + b
e = 2.71828
print(z)

SigmodFunction = 1 / (1 + math.pow(e,-z))
print(SigmodFunction)
# output della sigmod 2.77511564542893e-08

#------------------------------------
# VALORE GRANDE POSITIVO
#------------------------------------

w1 = 20
b1 = 30
x1 = 0.63
z1 = x1*w1 + b1
e1 = 2.71828
print(z1)

SigmodFunction1 = 1 / (1 + math.pow(e1,-z1))
print(SigmodFunction1)
#output della sigmod 1.0