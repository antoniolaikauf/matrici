from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
  
logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
print(model)