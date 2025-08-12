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
- head_dim: consiste nella dimensione di ogni teste di attenzione (Q, K, V) all'interno del blocco di attention.
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
- num_experts: parametro usato nell MLP nell'architettura Mixture of Experts
- experts_per_token: indica quanti experts sono usati per processare i token 
- initial_context_length: quantità di token che il modello può processare alla volta
- sliding_window: quantità di possibili token che il token può gurdare dietro se stesso duranante il meccanismo di attention 

classe Transformer è formata da:
- self.embedding: consiste nel Emebedding del token che va da **config.vocab_size**, fino a n_embedding, qua invece lo chiamano **config.hidden_size**
- self.block: consiste nella quantità di layer che compongono l'architettura, questo è dato da **config.num_hidden_layers**
- self.norm: consiste nella normalizzazione **dell'hidden_state** cosi che abbiano un valore coerente e non troppo diverso tra di loro 
- self.unembedding: converte **l'hidden_state** in logit cosi che ogni token abbia un valore per ogni token nel **vocab_size**, dopo che si applica la softamx questi diventano probabilità permettendo al modello di prevedere il prossimo token.

```
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

```

Blocco di TransformerBlock formato da:
- self.attn: consiste nel blocco di attention
- self.mlp: consiste nel blocco di **multi layer perception** o anche chiamato FFN **feed forward network**

```
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x

```

AttentionBlock non sarebbe la classica attention in cui ogni head utilizza le proprie Q, K ,V.
l'attention usata in gpt-oss è chiamata **Multy query attention**, il suo funzionamento consiste nel usare le K, V per un gruppo di head delle Q in questo caso le head delle Q sono 64 e invece quelle delle K, V sono 8, quindi un gruppo di 8 Q usera solo una delle K, V. <br>
Inoltre al posto di utilizzare la classica position embedding absolute o sinusoidale quella apprendibile utilizza la **Rotary position embedding**  
Blocco di AttentionBlock formato da:

- self.head_dim: consiste nella dimesione di ogni Q, K, V all'interno di ogni head.
- self.num_attention_heads: quantità di head per le Q.
- self.num_key_value_heads: quantità di head per le K, V.
- self.sliding_window: quanità di token a un singolo token può attendere o guardare indietro, questo permette di risparmiare risorse ma il compormesso è che in frasi lunghe i token che sono distanti tra di loro non si relazionano.
- self.sinks:
- self.norm: consiste nella normalizzazione **dell'hidden_state** cosi che abbiano un valore coerente e non troppo diverso tra di loro
- qkv_dim: dimensioni del vettore formato dalle Q, K, V. lunghezza 5120.
- self.qkv: linearizzazione standard di torch x*w + q. l'output di questa linearizzazione è di **qkv_dim**
- self.out: linearizzazione standard di torch x*w + q. l'output di questa linearizzazione è di **config.hidden_size**
- self.sm_scale: 
- self.rope: classe che permette al modello di capire la posizione dei token **Rotary position embedding**

```
   def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        # dimensione delle qkv 33.792
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )
```

SEE YOU LATER, THEY HAVE JUST RELEASE GPT-5
UPDATE 10/08/2025 GPT-5 WHEN AGI ??