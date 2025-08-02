from prepare import vocab_size

configGPT = {
    'n_head' : 12, # quantità di teste per ogni blocco
    'n_embd' : 768, # lunghezza vettore per ogni token
    'vocab_size' : vocab_size, # tot token 
    'n_layer' : 12, # quantità di layer della rete
    'block_size': 32 # quantità massima di token inseribili all'interno della rete
}