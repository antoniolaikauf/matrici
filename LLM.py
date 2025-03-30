# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # Carica tokenizer e modello
# tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
# model.generation_config.pad_token_id = tokenizer.eos_token_id

# keepGoing = True
# input_text = input()
# while (keepGoing):
#     inputs = tokenizer(input_text, return_tensors="pt")
#     attention_mask = inputs['attention_mask']

#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=attention_mask,
#         max_new_tokens=50,       # Genera 50 token nuovi, escluso l'input
#         do_sample=True,          # Abilita sampling per varietà
#         temperature=0.7,         # Controlla la casualità
#         top_p=0.9,              # Sampling più coerente
#         no_repeat_ngram_size=2,  # Evita ripetizioni di bigrammi
#         repetition_penalty=1.2   # Penalizza ripetizioni
#     )

#     # Decodifica e stampa
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if response.startswith(input_text):
#         response = response[len(input_text):].strip()
#     print(response)
#     keepGoing = input('premi K per continuare a giocare premi Q per smettere\n')
#     if keepGoing == 'Q': keepGoing = False 
#     else: input_text = input()


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("perplexity-ai/r1-1776", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("perplexity-ai/r1-1776", trust_remote_code=True)
model.generation_config.pad_token_id = tokenizer.eos_token_id

while True:
    
    input_text = input("Tu: ")
    if input_text.lower() == "exit":
        break

    # Tokenizza
    inputs = tokenizer(input_text, return_tensors="pt")
    attention_mask = inputs['attention_mask']

    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Bot:", response)