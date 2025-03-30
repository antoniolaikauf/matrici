from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carica tokenizer e modello
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
model.generation_config.pad_token_id = tokenizer.eos_token_id


input_text
# Input
input_text = "yes"
inputs = tokenizer(input_text, return_tensors="pt")
attention_mask = inputs['attention_mask']

# Genera risposta
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=attention_mask,
    max_new_tokens=50,       # Solo token nuovi, non include l’input
    do_sample=True,
    temperature=0.7,         # Controlla la casualità
    top_p=0.9,              # Sampling più coerente
    no_repeat_ngram_size=2,  # Evita ripetizioni di bigrammi
    repetition_penalty=1.2   # Penalizza ripetizioni
)

# Decodifica e stampa
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)