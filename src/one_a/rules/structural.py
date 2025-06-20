from one_a.utils import nlp, has_subject
from one_a.config import ALLOW_YOU_TO_VERBS

def ensure_subject_presence(sentence: str, subject: str = "I") -> str:
    doc = nlp(sentence)

    if has_subject(doc):
        return sentence

    is_question = sentence.strip().endswith("?")
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    aux = next((t for t in doc if t.dep_ == "aux" and t.head == root), None)

    new_tokens = []
    inserted = False

    for t in doc:
        if not inserted:
            if is_question and t.dep_ == "aux":
                new_tokens.append(t.text)
                new_tokens.append(subject)
                inserted = True
                continue
            if aux and t == aux or not aux and t == root:
                new_tokens.append(subject)
                inserted = True
        new_tokens.append(t.text)

    if not inserted:
        return sentence

    return " ".join(new_tokens)
    
def move_too_after_verb(sentence: str) -> str:
    doc = nlp(sentence)

    too_token = next((t for t in doc if t.text.lower() == "too" and t.pos_ == "ADV"), None)
    if not too_token or too_token.i >= len(doc) - 2:
        return sentence

    verb = next((t for t in doc[too_token.i + 1:] if t.pos_ in {"VERB", "AUX"}), None)
    if not verb:
        return sentence

    insert_at = len(doc)
    for t in doc[too_token.i + 1:]:
        if t.dep_ in {"dobj", "pobj"}:
            insert_at = t.i + 1
            break

    indices_to_remove = {too_token.i}
    if too_token.i + 1 < len(doc) and doc[too_token.i + 1].text == ",":
        indices_to_remove.add(too_token.i + 1)

    tokens_wo_too = [t.text for i, t in enumerate(doc) if i not in indices_to_remove]

    for removed_index in indices_to_remove:
        if removed_index < insert_at:
            insert_at -= 1

    if insert_at >= len(tokens_wo_too):
        tokens_wo_too.append("too")
    else:
        tokens_wo_too.insert(insert_at, "too")

    return " ".join(tokens_wo_too)

def fix_you_to_verb(sentence: str) -> str:
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    
    for i, tok in enumerate(doc[:-2]):
        if tok.text.lower() == "you":
            prev_verb = next((t for t in reversed(doc[:i]) if t.pos_ == "VERB"), None)
            next1 = doc[i + 1]
            next2 = doc[i + 2]
            if next1.text == "to" and next1.pos_ == "PART" and next2.pos_ == "VERB" and (
                not prev_verb or prev_verb.lemma_ not in ALLOW_YOU_TO_VERBS
            ):
                return " ".join(tokens[:i + 1] + [next2.text] + tokens[i + 3:])

    return sentence

def fix_missing_dash_between_clauses(sentence: str) -> str:
    doc = nlp(sentence)
    roots = [tok for tok in doc if tok.dep_ == "ROOT"]
    if len(roots) < 2:
        return sentence

    second_root = roots[1]
    clause_start = min(tok.i for tok in second_root.subtree)
    tokens = [tok.text for tok in doc]
    tokens.insert(clause_start, "—")
    return " ".join(tokens)