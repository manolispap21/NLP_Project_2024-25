from one_b.models import tok_t5, mod_t5, tok_pgs, mod_pgs, en2de, de2en, DEVICE

def paraphrase_t5(sentence, beams=10, temp=1.0, max_len=256):
    prompt = f'paraphrase: {sentence}'
    inputs = tok_t5(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    outputs = mod_t5.generate(
        inputs.input_ids,
        num_beams=beams,
        num_return_sequences=1,
        temperature=temp,
        max_length=max_len,
        early_stopping=True
    )
    return tok_t5.decode(outputs[0], skip_special_tokens=True)

def back_translate_en(sentence, do_sample=True, temp=0.9):
    de = en2de(sentence, do_sample=do_sample, temperature=temp)[0]['translation_text']
    back = de2en(de, do_sample=do_sample, temperature=temp)[0]['translation_text']
    return back

def paraphrase_pegasus(sentence, beams=5, returns=1, max_len=60, temp=1.5):
    batch = tok_pgs([sentence], truncation=True, padding='longest', max_length=512, return_tensors='pt').to(DEVICE)
    translated = mod_pgs.generate(
        **batch,
        max_length=max_len,
        num_beams=beams,
        num_return_sequences=returns,
        temperature=temp,
        early_stopping=True
    )
    return tok_pgs.decode(translated[0], skip_special_tokens=True)