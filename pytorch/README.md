# BOOK 

### gradient descent

è un algoritmo che permette di trovare il miglior set di parametri per un modello partendo da parametri randomici, per calcolare i gradianti prima bisogna calcolare la loss e questo si calcola facendo la differenza tra l'output del modello e quello che noi vorremmo ottenere 

$\frac{1}{n}$ * $\sum_{i=1}^{n} (yHat - y)^2$

nella formula si eleva alla seconda cosi che non escano valori negativi

Il gradiante è una derivata parziale perchè si calcola rispetto ad un singolo parametro. <br>
Il gradiante con la curva più ripida influenzerà maggiormente la loss function 