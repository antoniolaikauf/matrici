hyperparameter del modello rilasciato da openIA
```
@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
```

- vocab_size: Il vocab_size rappresenta il numero di token unici nel vocabolario del modello.
- hidden_size: Il hidden_size rappresenta la dimensione del vettore di rappresentazione di ogni token nelle fasi interne del modello.
- head_dim: consistenella dimensione di ogni teste di attenzione (Q, K, V) all'interno del blocco di attention.
- num_attention_heads: consiste nella quantità di heads in ogni blocco di attention, quindi il totale di head del modello viene dato da num_hidden_layers x num_attention_heads, questo parametro fa riferimento solo alle heads delle query.
- num_key_value_heads: Il num_key_value_heads rappresenta il numero di teste di attention per key e value, che in configurazioni come Grouped Query Attention è inferiore al numero di teste per le query.
- num_hidden_layers: numero  di layer all'interno dell'architettura del modello.
- swiglu_limit: parametro usato nella activation function nel blocco del MLP (multi-layer perceptron)
questo parametro serve per fare si che i valori dentro alla funzione di attivazione siano compresi tra un minimo ad un massimo. <br>
Nella funzione di attivazione, il parametro è usato come segue:    
x_glu = x_glu.clamp(min=None, max=limit)
x_linear = x_linear.clamp(min=-limit, max=limit)
quando il min è impostato a None allora non ci sarà un minimo in questo caso.
- intermediate_size: consiste nella dimensione dei parametri weight e bias nel blocco del MLP, dentro alla classe MLPBlock viene raddoppiato, quindi il vettore processato al suo interno diventerà il doppio.


classe Transformer è formata da:
- self.embedding che consiste nel Emebedding del token che va da **config.vocab_size**, fino a n_embedding, qua invece lo chiamano **config.hidden_size**




SEE YOU LATER, THEY HAVE JUST RELEASE GPT-5
UPDATE 10/08/2025 GPT-5WHEN AGI ??