from one_a.rules.structural import (
    ensure_subject_presence,
    move_too_after_verb,
    fix_you_to_verb,
    fix_missing_dash_between_clauses,
)
from one_a.rules.semantic import (
    fix_double_noun_phrases,
    correct_verb_noun_possessive,
    fix_temporal_prepositions,
    fix_as_my_phrase,
    fix_missing_with_in_conjunctions,
)
from one_a.rules.surface import (
    punctuation_fix,
    insert_missing_articles,
    fix_inner_capitalization,
    remove_redundant_subject_pronouns,
)

def rewrite_sentence(text):
    fixed = ensure_subject_presence(text)
    fixed = move_too_after_verb(fixed)
    fixed = fix_you_to_verb(fixed)
    fixed = fix_as_my_phrase(fixed)
    fixed = fix_missing_dash_between_clauses(fixed)
    fixed = fix_double_noun_phrases(fixed)
    fixed = correct_verb_noun_possessive(fixed)
    fixed = fix_temporal_prepositions(fixed)
    fixed = fix_missing_with_in_conjunctions(fixed)
    fixed = insert_missing_articles(fixed)
    fixed = remove_redundant_subject_pronouns(fixed)
    fixed = fix_inner_capitalization(fixed)
    fixed = punctuation_fix(fixed)
    return fixed