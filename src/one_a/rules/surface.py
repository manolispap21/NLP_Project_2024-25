from one_a.utils import nlp

def insert_missing_articles(text):
    doc = nlp(text)
    new_tokens = []
    i = 0

    while i < len(doc) - 3:
        tok1 = doc[i]
        tok2 = doc[i+1]
        tok3 = doc[i+2]
        tok4 = doc[i+3]
        if (
            tok1.pos_ == "ADP" and
            tok2.pos_ == "NOUN" and
            tok3.text.lower() == "and" and
            tok4.pos_ == "NOUN"
        ):
            new_tokens.extend([tok1.text, "the", tok2.text, tok3.text, "the", tok4.text])
            i += 4
            continue
        new_tokens.append(tok1.text)
        i += 1

    while i < len(doc):
        new_tokens.append(doc[i].text)
        i += 1

    return " ".join(new_tokens)

def punctuation_fix(text):
    return text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" ;",";")
    
def fix_inner_capitalization(text):
    """
    Μικραίνει το πρώτο γράμμα των λέξεων (εκτός της πρώτης) αν είναι κεφαλαίο.
    """
    doc = nlp(text)
    tokens = []
    for i, token in enumerate(doc):
        if i == 0:
            tokens.append(token.text)
        elif token.text == "i" or token.text == "I":
            tokens.append(token.text)
        elif token.text.istitle():
            tokens.append(token.text.lower())
        else:
            tokens.append(token.text)
    return " ".join(tokens)

def remove_redundant_subject_pronouns(sentence):
    doc = nlp(sentence)
    subj_heads = {}
    for token in doc:
        if token.dep_ == "nsubj":
            head = token.head
            subj_heads.setdefault(head, []).append(token)

    redundant_token_idxs = set()

    for head, tokens in subj_heads.items():
        nouns = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"}]
        prons = [t for t in tokens if t.pos_ == "PRON"]
        if nouns and prons:
            for p in prons:
                redundant_token_idxs.add(p.i)

    for i, token in enumerate(doc):
        if token.pos_ == "PRON" and token.dep_ == "nsubj" and i not in redundant_token_idxs:
            comma_count = 0
            noun_found = False
            j = i - 1
            while j >= 0:
                t = doc[j]
                if t.text == ".":
                    break
                if t.text == ",":
                    comma_count += 1
                if t.pos_ in {"NOUN", "PROPN"}:
                    noun_found = True
                j -= 1
            if comma_count >= 2 and noun_found:
                redundant_token_idxs.add(i)

    new_tokens = [t.text for i, t in enumerate(doc) if i not in redundant_token_idxs]
    fixed = " ".join(new_tokens)
    fixed = fixed.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return fixed