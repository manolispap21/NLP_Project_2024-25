# === Λεξικά μετατροπών ===

NOUN_TO_PARTICIPLE = {
    "delay": "delayed",
    "cancel": "cancelled",
    "fail": "failed",
    "freeze": "frozen",
    "close": "closed"
}

SUBJECT_TO_POSSESSIVE = {
    "i": "my",
    "you": "your",
    "he": "his",
    "she": "her",
    "it": "its",
    "we": "our",
    "they": "their"
}

# === Λέξεις που χρειάζονται κτητικό ===

NEEDS_POSSESSIVE = {"best", "all", "effort", "part", "support", "help"}

NEEDS_POSSESSIVE_VERBS = {"try", "give", "make", "offer", "do"}

# === Ρήματα που επιτρέπουν you + to + verb ===

ALLOW_YOU_TO_VERBS = {
    "want", "tell", "ask", "expect", "help", "allow", "force",
    "encourage", "advise", "remind", "permit", "order", "enable",
    "instruct", "persuade", "teach", "invite", "warn", "let"
}

# === Αφηρημένα ουσιαστικά που απαιτούν ειδική μεταχείριση ===

ABSTRACT_NOUNS = {"wish", "hope", "dream", "desire", "prayer", "intention", "thought"}

# === Χρονικά συμφραζόμενα ===

TEMPORAL_ADJECTIVES = {"recent", "last", "past", "previous", "next", "coming"}

TEMPORAL_NOUNS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours"}

WRONG_TEMPORAL_PREPOSITIONS = {"at", "on", "by"}