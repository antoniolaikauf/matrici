# ALGEBRA LINEARE 

### MATRICI
le matrici sono array contenenti numeri organizzati in row e colonne. <br>
Le dimensioni di una matrice viene data dalle row e colonne, una matrice con solo una row viene detta **row matrix**, invece una matrice con solo una colonna viene detta **column matrix** e una matrice con row e colonne uguali viene detta **square matrix**. <br>
Per selezionare un elemento dentro alla matrice si fa riferimento A[i,j] es A[1,3]

### ADDIZIONI CON MATRICI 
Per eseguire un addizioni con matrici le loro dimensioni devono essere uguali 

![](image/addMatrix.png)

### MOLTIPLICAZIONE SCALARE
una moltiplicazione tra un valore c e la matrice A è calcolata moltiplicando tutti gli elementi con c 

![](image/product.png)

### SOTTRAZIONE 
la sottrazione tra due matrici è come la addizione ma bisogna moltiplicare i valori di una matrice per -1

### MOLTIPLICAZIONE CON MATRICI
una moltiplicazione tra matrici si può avere solo se il numero di colonne della prima matrice è uguale al numero di row della seconda matrice 

![](image/product2.gif)

### HADAMART PRODUCT 
moltiplicazione tra matrici che prende due matrici di stesse dimensioni e ritorna una matrici di uguali dimensioni

![](IMAGE/hadamart-product.png)

### TRASPOSIZIONE 
Una trasposizione di una matrice A di dimensioni m x n è formata dalla trasformazione delle row in colonne e vice versa 

![](image/Trasposizione.png)

ci sono delle proprietà dei numeri che si utilizzano anche con le matrici 

- A + B = B + A
- (cA)<sup>T</sup>  = c(A<sup>T</sup>)
- A<sup>T</sup> + B<sup>T</sup> = (A + B)<sup>T</sup>
- (A<sup>T</sup>)<sup>T</sup> = A




# IMMAGINE VETTORI 

![](image/vettore.png)


# LIBRO

```markdown
modello tipo **perceptrons**/**percettore** prende vari input x1, x2, x3 e produce un singolo output, per calcolare l'output si introducono i **weights**, l'output di questa rete neurale è 0 o 1 ed è determinata se la somma delle weight nè maggiore di una threshold.
al cambiare delle weight e della treshold possiamo ottenere diversi modelli di processo decisionale
```
![](image/percettore.png)

[es. vedere libro link](http://neuralnetworksanddeeplearning.com/chap1.html)

ora la nostra equazione può essere riscritta come 

![](image/percettore2.png)

in cui tutti e i tre parametri sono vettori la x di lunghezza 1 x d_model (che d_model è 512) nel paper transformers la W sarà di 512 x 2048 e la bias di 1 x 2048 (queste lunghezze sono solo nella trasformazione lineare perchè nella seconda bisogna trasformare l'output di questa eqauzione a dimensioni di d_model)

quando la bias è molto alta è più facile che l'output sia 1 <br>

![](image/activation.png) 

![](image/percettoreEsempio.png)


per permettere alla rete di migliorare bisognerebbe fare piccoli cambiamenti alle weight e alle bias cosic he l'output cambi. <br> Ma bisogna stare attenti essendo che un cambiamento delle bias o weight potrebbe cambiare il cambiamennto della rete neurale. <br>
Per evitare questo problema allora si introduce un nuovo neurone chiamato **sigmoid neuron** che assomiglia molto al percettore ma gli input possono avere valori tra 0 e 1 invece di 0 o 1 come nel percettore e l'output non è 0 o 1 **ma un valore tra 0 e 1** <br>

![](image/sigmodFunction.png) <br>

![](image/textSigmodFunction.png)

la sigmod function in alcuni dsa l'output tra un valore 0 e 1 e quando z tende ad essere un valore negativo molto grande l'output della funzione sarà 1 invece quando la z assume un valore positivo molto grande allora l'output della funzione sarà 0. <br>
In questi due casi la funzione assomiglia molto al percettrone.

### Forme Funzioni

![](image/functionShape.png)


Si può notare che la funzione sigmod è una fuznione più 'levigata' rispetto al percettrone. <br>

Queste due funzioni sono chiamate activation function 
(Una funzione di attivazione in una rete neurale è una funzione matematica che determina se un neurone deve essere attivato in base ai suoi segnali di ingresso.)
  

La rete neurale è composta da input layer in cui ci sono gli input x1, x2 ,x3. L'hidden layer significas 'non un input e non un output', invece l'output layer è lo stato in cui fuoriesce l'output <br>

![](image/neuralNetwork.png)

Una rete in cui l'output del layer viene usato come input per il prossimo layer viene chiamato **feedFoward** 

<br> <br>

questa rete da un output in binario della rappresentazione di un numero 
es. 2 si illuminano 0010

![](image/neuralNetwork2.png)

In questo caso, si vuole scegliere un set di pesi (weights) e bias tale da illuminare i bit necessari per rappresentare il numero. Nel vecchio output layer, si illumina un neurone in base all’input: se l’input è 2, si illumina il terzo neurone e la sequenza diventa [0, 0, 0.99, 0, 0, 0, 0, 0, 0, 0].
Per scegliere i pesi, si fa in modo che siano abbastanza grandi da far sì che la funzione sigmoide restituisca 0.99 (e non 0.01). Ad esempio:  
- Il primo neurone, che rappresenta il bit più significativo, ha pesi [0, 0, 0, 0, 0, 0, 0, 0, 10, 10];  

- Il secondo ha [0, 0, 0, 0, 10, 10, 10, 10, 0, 0];  

- Il terzo ha [0, 0, 10, 10, 0, 0, 10, 10, 0, 0]; 

- Il quarto ha [0, 10, 0, 10, 0, 10, 0, 10, 0, 10].

Questo avviene perché, per esempio, il primo neurone deve illuminarsi solo quando l’input è 8 o 9, dato che le loro rappresentazioni binarie sono 1000 (8) e 1001 (9). La bias, invece, viene scelta uguale a -5. <br>
Con un input pari a 2, si dovrebbe illuminare il terzo neurone del quarto strato. Nei pesi del terzo neurone, [0, 0, 10, 10, 0, 0, 10, 10, 0, 0], si nota che la posizione 2 (che corrisponde alla cifra 2) ha un peso di 10. Così, quando si calcola la somma pesata con l’equazione xi*wi + b
, si moltiplica [0, 0, 0.99, 0, 0, 0, 0, 0, 0, 0] per i pesi del terzo neurone [0, 0, 10, 10, 0, 0, 10, 10, 0, 0]. Il risultato è 10⋅0.99+(−5)=9.9−5=4.9.<br>
Inserendo 4.9 nella funzione sigmoide, si ottiene circa 0.99, permettendo di illuminare il bit 3. Per gli altri neuroni, invece, il risultato è 0.
Ad esempio, moltiplicando [0, 0, 0.99, 0, 0, 0, 0, 0, 0, 0] per i pesi del secondo neurone [0, 0, 0, 0, 10, 10, 10, 10, 0, 0], si ottiene 0; quindi z=0+(−5)=−5z = 0 + (-5) = -5z = 0 + (-5) = -5
. Messo nella sigmoide, -5 restituisce circa 0.01, quindi il secondo neurone non si illumina. Lo stesso vale per gli altri neuroni.

Ovviamente il processo di trovare le weight e le bias avviene tramite una funzione 

![](image/constFunction.png)

- w sono le weight 
- b è la bias
- n sono la quantità di esempi usati nel training 
- a è l'output per ogni xn quindi nel contesto di pirma sarebbe a=[0.01, 0.01, 0.99, 0.01]
- y(x) è il valore che il modello dovrebbe produrre per un dato input ( x ) 
- la somma è su tutti gli input nel trainig set 

- ∣∣y(x)−a∣∣<sup>2</sup>=
(y(x)1−a1)<sup>2</sup>+(y(x)2−a2)<sup>2</sup>+...+ (y(x)−ax)<sup>2</sup>


se questa funzione raggiunge presso che 0 allora si sono trovate un ottimo set di bias e weight se invece raggiunge un numero grande allora l'algoritmo non sta andando bene questa perchè nella loss function più il valore è basso e migliore è il nostro algoritmo.
Questo perchè più i valori di a che sarebbe l'output del nostro modello e i valori di f(x) sono uguali la funzione tenderà a 0 

es. y(x)=[0, 0, 1, 0],a=[0.01, 0.01, 0.99, 0.01] <br>
y(x) − a=[0 − 0.01, 0 − 0.01, 1 − 0.99 , 0 − 0.01]=[−0.01, −0.01, 0.01, −0.01] <br>
∣∣y(x)−a∣∣<sup>2</sup>=(−0.01)<sup>2</sup>+(−0.01)<sup>2</sup>+(0.01)<sup>2</sup>+(−0.01)<sup>2</sup>=4 * (0.01)<sup>2</sup>=0.0004

La tecnica **gradian descent** viene usata per trovare weight e bias che fanno ottenere una loss functioin bassa 

per capire come cambia la loss function se is modificano i parametri si utilizza questa equazione, quindi dirà una stima di quanto cambierà la loss se aggiorni i pesi  <br>
![](image/funzioneCambio.png)

- Rappresenta il cambiamento approssimativo della funzione di costo ( C ) quando modifichi i parametri v1 e v2  di una quantità Δv1 e Δv2 <br>
quindi il gradiente (∇C) sarebbe $\frac{∂C}{∂v1}$ e $\frac{∂C}{∂v2}$. Queste $\frac{∂C}{∂v1}$ e $\frac{∂C}{∂v2}$ sarebbero le derivate della loss function rispetto a W e la derivata della loss function rispetto alla bias in cui la derivata sarebbe **una funzione che rappresenta il tasso di cambiamento di una data funzione rispetto a una certa variabile, vale a dire la misura di quanto il valore di una funzione cambi al variare del suo argomento. Più informalmente, la derivata misura la crescita (o decrescita) che avrebbe una funzione in uno specifico punto spostandosi di pochissimo dal punto considerato.** <br>
![](image/derivata.png)  

- Δv1 Δv2 sono i passi che si fanno per far si che si 'scenda' e quindi si ottenga una loss function con un valore sempre più piccolo <br>
Δv1 = $\frac{∂C}{∂v1}$ * -η  Δv2 = $\frac{∂C}{∂v2}$ * -η
- -η sarebbe il learning rate (è un valore piccolo) inoltre è il learning rate che dice di quando bisogna 'scendere' nella funzione della gradiant descent. 
Se il learning rate è troppo piccolo ci vorranno più step per raggiungere il minimo, invece se il learning rate è troppo grande si possono avere **divergenze** in cui l'algoritmo non raggiunge il minimo ma, 

quindi alla fine l'eqauzione si può riscrivere come <br><br>
![](image/gradianteEquazione.png)<br>
 in cui ΔC consiste nel gradiante ∇C ( sarebbe $\frac{∂C}{∂v1}$ e $\frac{∂C}{∂v2}$  ) moltiplicato con il prodotto tra il gradiante ∇C e il learning rate 

P.S (i gradianti del input si calolano ma non si cambiano essendo che non sono utili perchè gli input non si cambiano)

ovviamente si deve andare nella direzione −∇C perchè se no la loss aumenterebbe 

- prima si calcola la loss function C
-  una volta calcolato si usa l'equazione ΔC in cui ci dice di quanto dovrebbe diminuire l'errore con il nuovo set di weight i bias 
- si calcola ancora una volta C con le weight e bias modificate del passaggio 2 
- si verifica che il pssaggio tre sia corretto facendo il passaggio 1 meno il 3 e dovrebbe risultare il risultato del passaggio 2 

**l'idea principale del gradiant decsent è quella di trovare le weight e le bias che permettono di trovare il minimo nella loss function**

quindi le nuove weight si calcolano facendo la differenza tra le weight attuali - lo step che in questo caso rappresenterebbe -η∇C. <br><br>
![](image/newParameters.png)

### stochastic gradient descent

volendo se il set di input è troppo grande si sceglie un batch di grandezza minore rispetto al dataset e lo si allena su questo, ovviamente il dataset dovrebbe avere molti esempi per far si che funzioni questo <br> <br>
![](image/batch.png)
<br>
m consiste nel batch e la grandezza di m deve risultare minore di n. m < n. <br>
Una cosa obbligatoria da fare è che gli esempi che sono utilizzati per creare m devono essere scelti randomicamentre dal dataset. <br>
Il stochastic gradient descent funziona grazie alla **legge dei grandi numeri** 

```markdown
Secondo la legge dei grandi numeri è ragionevolmente sicuro che la media, che determiniamo a partire da un numero sufficiente di campioni, sia sufficientemente vicina alla media vera, ovvero quella calcolabile teoricamente
```

ovviamente questo deve avvenure più volte in modo tale che le weight e le bias vengano aggiornate e una volta finito il dataset completo si rinizia da 0 ma senza usare gli stessi epoch utilizzati precedentemente. <br>
es [1,2,3,4,5,6] dataset, primo batch [3,5] secondo batch [6,1] terzo batch [4,2]  e quersto processo viene chiamato epoch
N.B in un ciclo possono uscire più volte gli stessi esempi essendo che è stocastico e quindi randomico es [1,2,3,4,5,6] dataset, primo epoch [3,5] secondo [6,1] terzo[4,6] può succedere 

Ovviamente i vantaggi del **stochastic gradient descent** è che è più veloce rispetto al **gradient descent** se il dataset è grande 
e anche gli aggiornamenti dei parametri delle weight e bias avviene più veloce essendo che vengono aggiornati tot alla volta e non tutti insieme e se si processano milioni o anche migliardi di parametri in una volta sola sarebbe difficile per l'hardware caricarli e processarli tutti in uno.

GD:<br>
Calcoli il gradiente su tutte le 1.000.000 immagini.

se un aggiornamento richieda 600 secondi (10 minuti) a causa del volume di dati.

Dopo 10 epoche (10 aggiornamenti), hai speso  10 × 600= 6.000 secondi (100 minuti).
<br><br><br>
SGD (batch size = 32):<br>
Ogni mini-batch (32 immagini) richiede 0,02 secondi.

un epoch ha $\frac{1.000.000}{32}$  = 31.250 che sarebbero la quantità dei mini-batch, 
quindi 31.250 × 0,02 = 625 secondi che sarebbero 10 minuti

Ma il modello potrebbe raggiungere una buona loss dopo solo 5 epoche, quindi 5x625 = 3125 secondi che in minuti risulterebbe 52 

**si può presuppore che con SGD si ottenga una buona loss function solo dopo 5 epoch perchè con sdg si tropva più velocemenete il minimo della loss function**

una correzzione da fare è che molto probabilmente mentre si legge si può pensare che faccia tutto il gradiant descent ma in verità ci sarebbe anche la **backPropagation**. <br>
La backpropagation consiste nel calcolare i gradianti e quindi gli step che permettono di diminuire la loss, invece la gradiant descent consiste nell'aggiornamento dei parametri 
quindi questi due algoritmi lavorano insieme 

### BACKPROPAGATION

l'obbiettivo della loss function è quella di calcolare le derivate della funzione di loss con rispettoa a qualsiasi weight e bias della rete 