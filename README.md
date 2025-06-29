# NLP_Project_2024-25

Απαλλακτική εργασία για το μάθημα **Επεξεργασία Φυσικής Γλώσσας**, Τμήμα Πληροφορικής – Πανεπιστήμιο Πειραιώς.

## Περιγραφή

Το έργο αυτό αφορά την εφαρμογή τεχνικών Επεξεργασίας Φυσικής Γλώσσας για την ανακατασκευή προτάσεων, τον υπολογισμό σημασιολογικής ομοιότητας και την αξιολόγηση διαφορετικών pipelines αναπαραγωγής. Περιλαμβάνει:
- επανεγγραφή φυσικής γλώσσας μέσω custom και αυτόματων μοντέλων,
- χρήση word embeddings και cosine similarity για μέτρηση νοηματικής απόστασης,
- αξιολόγηση μοντέλων σε συμπλήρωση νομικών προτάσεων.

## Δομή Project

NLP_Project_2024-25/
├── data/
│   ├── inputs/               # Είσοδοι (.txt)
│   └── outputs/              # Έξοδοι (.json, .csv, .png)
├── src/
│   ├── one_a/                # 1Α – κανόνες ανακατασκευής
│   ├── one_b/                # 1Β – paraphrasing με transformers
│   ├── two/                  # 2 – σημασιολογική ανάλυση
│   └── bonus/                # Bonus – masked clause completion
├── main.py                   # Entry point του project
├── pyproject.toml            # Poetry dependencies
├── poetry.lock               # Locked versions
└── README.md                 # Αυτό το αρχείο

## Οδηγίες Εγκατάστασης:

1. Κατεβάστε ή κλωνοποιήστε το project.

2. Εγκαταστήστε το poetry από την επίσημη σελίδα: https://python-poetry.org/docs/

3. Ανοίξτε τερματικό (ως διαχειριστής) στον φάκελο του project και εκτελέστε:
	poetry install

4. Για να εκτελέσετε τον κώδικα:
	poetry run python main.py
	
## Περιγραφή Λειτουργιών:

1Α – Ανακατασκευή Προτάσεων:
Διαβάζει δύο προτάσεις από τα sentence1.txt και sentence2.txt, και εφαρμόζει κανόνες (δομικούς, σημασιολογικούς, επιφανειακούς) για την ανακατασκευή τους. Οι ανακατασκευασμένες προτάσεις αποθηκεύονται στο αρχείο outputs_1a.json.

1Β – Παραφράσεις Κειμένων:
Δημιουργεί παραφράσεις για τα text1.txt και text2.txt με τρεις μεθόδους: T5, Pegasus και backtranslation. Αποθηκεύονται στο αρχείο outputs_1b.json.

2 – Σημασιολογική Ανάλυση:
Φορτώνει τις αρχικές και ανακατασκευασμένες εκδοχές των προτάσεων από το αρχείο outputs.json, υπολογίζει ενσωματώσεις λέξεων με το μοντέλο all-mpnet-base-v2 και μετρά cosine similarity μεταξύ τους. Τα αποτελέσματα συνοψίζονται σε πίνακες και οπτικοποιούνται.

Bonus – Masked Clause Completion:
Συμπληρώνει ημιτελείς προτάσεις με μάσκες λέξεων μέσω transformers και κάνει σημασιολογική και συντακτική σύγκριση, με χρήση της βιβλιοθήκης stanza. Τα αποτελέσματα είναι: masked_completion_outputs.json, mask_similarity_comparison.csv, syntax_analysis.csv.

## Μενού Εκτέλεσης

Το `main.py` περιλαμβάνει διαδραστικό μενού το οποίο επιτρέπει στον χρήστη να εκτελέσει τις διάφορες λειτουργίες της εργασίας χωρίς να χρειάζεται να γράψει παραμέτρους γραμμής εντολών.

## Επιλογές Μενού:

| Επιλογή |                                  Περιγραφή                                             |
|---------|----------------------------------------------------------------------------------------|
|   `1`   | Εκτελεί τα **1A και 1B**: ανακατασκευή προτάσεων και παραφράσεις κειμένων.             |
|   `2`   | Εκτελεί το **2**: υπολογιστική ανάλυση με cosine similarity και embeddings.            |
|   `3`   | Εκτελεί το **Bonus**: συμπλήρωση ελλιπών νομικών προτάσεων.                            |
|   `4`   | Εκτελεί συντακτική ανάλυση των αποτελεσμάτων του Bonus.                                |
|   `5`   | Συγκρίνει τις προβλέψεις του Bonus με το ground truth του Αστικού Κώδικα.              |
| `exit`  | Τερματίζει το πρόγραμμα.                                                               |

Το πρόγραμμα λειτουργεί επαναληπτικά μέχρι να δοθεί η εντολή `exit`, επιτρέποντας πολλαπλές εκτελέσεις χωρίς επανεκκίνηση.

## Αρχεία & Περιγραφή Κώδικα

|             Αρχείο                |                               Περιγραφή                                   |
|-----------------------------------|---------------------------------------------------------------------------|
| `main.py`                         | Entry point – Επιλέγει και εκτελεί τις διάφορες λειτουργίες               |
| `src/bonus/compare_similarity.py` | Συγκρίνει αποτελέσματα συμπλήρωσης `[MASK]` με ground truth               |
| `src/bonus/masked_completion.py`  | Συμπλήρωση νομικών προτάσεων με μοντέλα όπως BERT                         |
| `src/bonus/syntax_analysis.py`    | Συντακτική ανάλυση (π.χ. POS tags, sub/verb detection) για νομικό κείμενο |
| `src/one_a/config.py`             | Λεξικά και σταθερές για κανόνες ανακατασκευής (1A)                        |
| `src/one_a/pipeline.py`           | Κύρια συνάρτηση `rewrite_sentence()` – εφαρμόζει όλους τους κανόνες       |
| `src/one_a/utils.py`              | Βοηθητικά (π.χ. `has_subject`, φόρτωση κειμένου)                          |
| `src/one_a/rules/semantic.py`     | Σημασιολογικοί κανόνες – π.χ. κτητικά, χρονικά συμφραζόμενα               |
| `src/one_a/rules/structural.py`   | Δομικοί κανόνες – π.χ. τοποθέτηση υποκειμένου, "too after verb"           |
| `src/one_a/rules/surface.py`      | Επιφανειακοί κανόνες – π.χ. στίξη, πεζοκεφαλαία, αφαίρεση διπλοτύπων      |
| `src/one_b/models.py`             | Φορτώνει μοντέλα T5, Pegasus και tokenizer                                |
| `src/one_b/paraphrasers.py`       | Ορίζει pipelines για παραφράσεις (pipeline A, B, C)                       |
| `src/one_b/processing.py`         | Οργανώνει την εκτέλεση παραφράσεων για το 1B                              |
| `src/two/pipeline.py`             | Υλοποιεί υπολογισμό cosine similarity & word embeddings                   |
| `src/two/runner.py`               | Εκτελεί τη ροή του 2 – embedding, σύγκριση, PCA                           |

## Αποτελέσματα:
Όλα τα παραγόμενα αρχεία αποθηκεύονται στον φάκελο data/outputs, σε μορφή .json, .csv ή .png.

## Εξαρτήσεις:
Python <3.13,>=3.10
transformers
sentence-transformers
nltk
stanza
scikit-learn
matplotlib
(Πλήρης λίστα στο pyproject.toml)
