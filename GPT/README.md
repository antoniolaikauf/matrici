repo per la creazione di GPT mini stile karpathy, prenderò spunto dalla sua repo, probabilmente non sarà molto avanzata ma penso che il miglior modo per imparare le cose è fare le cose.
prenderò spunto anche dalla repo miniGPT, la rete neurale sarà la stessa ma con tutti i parametri di GPT2 del paper e ovviamente anche il tiktokenizer chge sarebbe il cambiamento principale

gpt ha un ratio di compressione di 1:3 quindi se si prende 1000 lettere saranno 300 token circa

```markdown
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encoder(data)
print(data[:10])
```
(P.S si potrebbe creare una rete che riconosce l'immagine e riesce fare uno schema, ma questo per un futuro)
