import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration,
    pipeline
)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- T5 Paraphraser ---
tok_t5 = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
mod_t5 = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws').to(DEVICE)

# --- Pegasus Paraphraser ---
pg_model_name = 'tuner007/pegasus_paraphrase'
tok_pgs = PegasusTokenizer.from_pretrained(pg_model_name)
mod_pgs = PegasusForConditionalGeneration.from_pretrained(pg_model_name).to(DEVICE)

# --- Backtranslation Pipelines ---
en2de = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de', device=0 if torch.cuda.is_available() else -1)
de2en = pipeline('translation_de_to_en', model='Helsinki-NLP/opus-mt-de-en', device=0 if torch.cuda.is_available() else -1)
