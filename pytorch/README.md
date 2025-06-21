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

### backward

backward() è il metodo che serve per calcolare i gradianti e aggiornarli, **i gradianti vengono aggiornati solo ai tensor che hanno requires_grad=True e anche ai tensor con cui si fanno operazione**, y_hat = train_x * w1 + b1 y_hat avrà requires_grad=true anche se non si specifica nel codice, per vedere il gradiante si usa **.grad**

## Optimizer

Optimizer servono per aggiornare i parametri, in pytorch ci sono vari Optimizer come **SGD** (Stochastic gradiant descent) **Adam**, **Momentum** ecc... <br> ![](https://cs231n.github.io/assets/nn3/opt2.gif) 


N.b se il tensor si trova sulla GPU non si può trasformarlo in un array numpy con .numpy, prima bisogna portarlo sulla CPU e dopo trasformalo in un array di numpy.
si ha un simile problema anche con tensor che hanno i gradianti e quindi prima bisogna usare il metodo .detach() e dopo .numpy()

Module è la classe base to torch, se si volesse creare un proprio modello allora bisognerebbe che il proprio modello erediti le altre proprietà del modello di base 

```markdown
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super(ManualLinearRegression, self).__init__()
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

    def foward(self, x):
        return x * self.w + self.b
```

Per ottenere tutte le Funzionalità del modello bisogna settare il .train() che permetterà di ottenere caratteristiche come il dropout o la batch normalization, permette di calcolare i gradianti.

## Modello

Il modello in pytorch ha due stati eval() valutazione e train() addrestramento. <br>
Nella modalità di train() il modello abilità specifici layer come il **dropout** e la **noramlization**. Nel dropout alcune connessioni vengono casualmente disabilita (con una probabilità specifica) e la noramlization in cui vengono normalizzati i dati in base alle media e varianza. Invece eval() il layer del **dropout** non viene attivato e il layer della **noramlization** viene attivato ma mai aggiornato. <br>
Il modello può essere salvato per evitare che si perdano i dati dopo tot tempo che si sta allenando e questo avviene tramite **torch.save** e si salvano i parametri che si vuole epoch, lo stato del modello (parametri), la loss ecc.., per caricarlo invece si usa **torch.load** e l'optimizer e il modello si caricano con **load_state_dict()** e ricordarsi ogni volta di settarlo con train() questo se si vuole ancora allenarlo, invece se si vuole fare predizione si setta in **eval()**




