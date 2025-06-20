from one_a.utils import nlp, get_subject_possessive
from one_a.config import NOUN_TO_PARTICIPLE, ABSTRACT_NOUNS, SUBJECT_TO_POSSESSIVE, NEEDS_POSSESSIVE, NEEDS_POSSESSIVE_VERBS, TEMPORAL_ADJECTIVES, TEMPORAL_NOUNS, WRONG_TEMPORAL_PREPOSITIONS

def correct_verb_noun_possessive(text):
    doc = nlp(text)
    new_tokens = []
    i = 0

    while i < len(doc):
        token = doc[i]
        if token.lemma_ in NEEDS_POSSESSIVE_VERBS and i + 1 < len(doc):
            next1 = doc[i + 1]
            next2 = doc[i + 2] if i + 2 < len(doc) else None
            if next1.text.lower() in NEEDS_POSSESSIVE:
                if next1.tag_ == "PRP$":
                    new_tokens.extend([token.text, next1.text])
                    if next2: new_tokens.append(next2.text)
                    i += 3 if next2 else 2
                    continue
                possessive = get_subject_possessive(token)
                new_tokens.extend([token.text, possessive, next1.text])
                if next2 and next2.text.lower() not in NEEDS_POSSESSIVE:
                    i += 2
                elif next2:
                    new_tokens.append(next2.text)
                    i += 3
                else:
                    i += 2
                continue
        new_tokens.append(token.text)
        i += 1

    return " ".join(new_tokens)

def fix_temporal_prepositions(text):
    doc = nlp(text)
    new_tokens = []
    i = 0

    while i < len(doc) - 2:
        token1 = doc[i]
        token2 = doc[i+1]
        token3 = doc[i+2]
        if (
            token1.text.lower() in WRONG_TEMPORAL_PREPOSITIONS and
            token2.text.lower() in TEMPORAL_ADJECTIVES and
            token3.text.lower() in TEMPORAL_NOUNS
        ):
            new_tokens.extend(["in", token2.text, token3.text])
            i += 3
        else:
            new_tokens.append(token1.text)
            i += 1

    while i < len(doc):
        new_tokens.append(doc[i].text)
        i += 1

    return " ".join(new_tokens)

def fix_as_my_phrase(sentence: str) -> str:
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    
    for i in range(len(doc) - 3):
        if (
            doc[i].text.lower() == "as" and
            doc[i].dep_ == "prep" and
            doc[i+1].text.lower() == "my" and
            doc[i+2].pos_ in {"ADJ", "NOUN"} and
            doc[i+3].lemma_ in ABSTRACT_NOUNS and
            doc[i+3].dep_ == "pobj"
        ):
            adj = doc[i+2].text if doc[i+2].pos_ == "ADJ" else ""
            noun = doc[i+3].lemma_
            new = tokens[:i] + ["it", "is", "my"]
            if adj:
                new.append(adj)
            new.append(noun)
            return " ".join(new + tokens[i+4:])

    return sentence

def fix_missing_with_in_conjunctions(text):
    doc = nlp(text)
    new_tokens = []
    i = 0
    skip_indices = set()

    while i < len(doc) - 3:
        token1 = doc[i]
        token2 = doc[i+1]
        token3 = doc[i+2]
        token4 = doc[i+3]
        if (
            token1.pos_ in {"ADJ", "VERB"} and
            token2.text.lower() == "and" and
            token3.pos_ in {"DET", "ADJ"} and
            token4.pos_ == "NOUN"
        ):
            new_tokens.extend([token1.text, "and", "with", token3.text, token4.text])
            skip_indices.update({i, i+1, i+2, i+3})
            i += 4
            continue
        if i not in skip_indices:
            new_tokens.append(doc[i].text)
        i += 1

    while i < len(doc):
        if i not in skip_indices:
            new_tokens.append(doc[i].text)
        i += 1

    return " ".join(new_tokens)

def fix_double_noun_phrases(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    new_tokens = []
    i = 0

    while i < len(doc) - 1:
        token1 = doc[i]
        token2 = doc[i+1]
        if token1.pos_ == "NOUN" and token2.pos_ == "NOUN":
            if token2.lemma_ in NOUN_TO_PARTICIPLE:
                if i == 0 or doc[i-1].lower_ not in ["a", "an", "the"]:
                    new_tokens.append("a")
                new_tokens.append(token1.text)
                new_tokens.append(NOUN_TO_PARTICIPLE[token2.lemma_])
                i += 2
                continue
        new_tokens.append(token1.text)
        i += 1

    if i == len(doc) - 1:
        new_tokens.append(doc[-1].text)

    return " ".join(new_tokens)