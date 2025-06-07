# BOOK 

### Gradient descent

Gradient descent è un algoritmo che permette di trovare il miglior set di parametri per un modello partendo da parametri randomici, per calcolare i gradianti prima bisogna calcolare la loss e questo si calcola facendo la differenza tra l'output del modello e quello che noi vorremmo ottenere. Il gradiante ti dice l'impatto che ha sulla loss

$\frac{1}{n}$ * $\sum_{i=1}^{n} (yHat - y)^2$

nella formula si eleva alla seconda cosi che non escano valori negativi

Il gradiante è una derivata parziale perchè si calcola rispetto ad un singolo parametro. <br>
Il gradiante con la curva più ripida influenzerà maggiormente la loss function. <br>
Il gradiante è molto importante essendo che è grazie a questo che la rete 'impara' quindi più è piccolo il gradianete e più gli aggiornamenti saranno irrilevanti arrivando ad un punto in cui la rete smette di imparare.

### Backpropagation

Backpropagation è un algoritmo che funzione calcolando la loss function rispetto ad ogni peso, calcolando il gradiante di ogni layer, la **Backpropagation** non è altro che un **gradiant descent incatenato**

### Learning rate

Il learning rate è un hyper parameter che dobbiamno applicare al gradiante. Il learning rate decide la grandeza del passo. <br>
Un learning rate troppo basso potrebbe volerci troppo per allenare il modello e quindi farci perdere tempo e soldi, invece un learning rate troppo alto non potrebbe portarci ad un minimo locale e quindi **divergere**. Finchè la curva è rotonda e bella prima o poi otterrai il minimo anche con un learning rate elevato ma con modelli più avanzati la curva non è rotonda e quindi non riusciresti a trovare il minimno. 
più la curva è ripida e più il learning rate deve essere basso, se invece è piatta 'flat' allora si può scegliere un learning rate più alto    

### LA COMBINAZIONE TRA  LEARNING RATE E GRADIANT PERMETTE L'AGGIORNAMENTO DEI PARAMETRI


### Epoch

Un Epoch sarebbe una sessione di allenamento in cui tu esegui ogni esercizio, in machine learning un epoch consiste di utilizzare tutti i dati nei passaggi: 
- Foward
- Calcolo la loss
- Backpropagation
- gradiant descent 


# Pytorch

In pytorch esistono i **Tensor** che sarebbero strutture vettori, matrici o array di dimensioni superiori, i tensor hanno dimensioni che possono variare, possono avere da 0 a più dimensioni con size() e volendo anche cabiare dimensioni con view()

le operazioni con i tensor possono essere eseguite con una **CPU** (computer process unit) o anche con una **GPU** (graphics process unit), con la CPU i valori vengono salvati sulla memoria principale invece con la GPU i dati vengono salvati sulla memoria dell GPU. <br>  Le GPU sono perfette per processare grandi quantità di dati essendo che sono ottime per l'elaborazione in parallelo e possono fare migliaia di operazioni in una volta sola. <br>
Per inviare i tensor al device che si vuole allora bisogna usare il metodo **to()** gpu_tensor = torch.as_tensor(x_train).to(device) 