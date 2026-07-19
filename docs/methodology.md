# Methodology

This document explains the *why* behind the design decisions in this
project, not just the *what*. It's written for anyone reviewing the repo
(a recruiter, an instructor, a future me) who wants to understand the
reasoning, not just read the code.

## 1. Problem framing

The task brief specifies text classification into positive / negative /
neutral across reviews, social media, and news. The single biggest
decision this project makes is to **treat "domain" as a first-class
variable** rather than a detail to abstract away. Sentiment is not domain
independent: "this is sick" is positive on Twitter and alarming in a
product review; "in line with expectations" is neutral-to-positive in
financial text and flat in a review. A model trained on one domain and
blindly applied to another routinely loses accuracy — this project
measures that loss instead of ignoring it.

## 2. Dataset selection

See `docs/dataset_description.md` for the full breakdown. In short: one
dataset per domain named in the brief (reviews, social media, news),
chosen so that:
- all three arrive with clean, well-documented labels or ratings (no
  hand-labeling required)
- combined they cover a wide range of text formality, length, and
  sarcasm/idiom density
- all three are small enough to process with classical ML on a laptop

## 3. Label unification

Amazon star ratings (1–5) are mapped to three classes (see table in
`dataset_description.md`) rather than treated as ordinal regression,
so that all three domains share exactly the same three-class target and
can be combined, compared, and cross-evaluated directly.

## 4. Preprocessing choices

- **Lowercasing, URL/mention/HTML stripping:** standard noise removal,
  necessary because Twitter and Amazon text both contain markup/links
  that carry no sentiment signal.
- **Hashtag handling:** the `#` symbol is stripped but the word is kept
  (`#terrible` → `terrible`) since hashtag words are often sentiment-
  bearing, unlike `@mentions`, which are dropped entirely (they're
  usually addressee names, not sentiment).
- **Digit removal:** numbers rarely carry sentiment on their own, and
  removing them keeps the TF-IDF vocabulary focused on words.
- **Stopword removal + lemmatization (NLTK):** reduces vocabulary size
  and sparsity for the TF-IDF stage without needing a neural tokenizer.

## 5. Feature engineering: why TF-IDF, not embeddings

TF-IDF with unigrams+bigrams was chosen over word embeddings or a
transformer encoder for three reasons:
1. **Interpretability** — with a linear model on top, you can read off
   exactly which n-grams push a prediction toward each class
   (`visualization.py::plot_top_features`). That's valuable for a
   portfolio project meant to demonstrate understanding, not just
   output a number.
2. **Sufficiency** — on datasets of this size (tens of thousands of
   rows), TF-IDF + a linear classifier is a well-established strong
   baseline that often performs within a few points of more complex
   models, while training in seconds instead of minutes/hours.
3. **Reproducibility** — no GPU, no large pretrained weights to download;
   the entire pipeline runs on a laptop or in CI in well under a minute.

## 6. Model selection

Two linear classifiers are trained on identical features — Logistic
Regression and Linear SVM — and compared by **macro-F1**, not accuracy.
Macro-F1 was chosen deliberately: financial text in particular is
dominated by the neutral class, and accuracy alone would let a model
that mostly predicts "neutral" look artificially strong. Macro-F1 weighs
all three classes equally, so it directly penalizes a model that ignores
the minority classes. `class_weight="balanced"` is also used on both
models for the same reason.

## 7. The core analytical contribution: cross-domain evaluation

`src/cross_domain_eval.py` trains a fresh model on each domain in
isolation and evaluates it against every other domain's held-out test
set, producing a 3×3 macro-F1 matrix. The diagonal is the in-domain
baseline; off-diagonal cells show the generalization gap. This is then
compared against a single model trained on all three domains pooled
together, to test whether combining domains during training recovers
generalization loss (multi-domain training as a regularizer) or whether
domain-specific patterns dilute each other. Whatever the result is on
your run, report it honestly in the README — a negative result ("pooling
domains didn't help much for X") is still a real, defensible finding and
demonstrates analytical maturity, not a failure.

## 8. Evaluation

Final metrics reported: overall accuracy, overall macro-F1, full
per-class precision/recall/F1 (`classification_report`), a confusion
matrix, and a per-domain accuracy/macro-F1 breakdown on the combined
model's held-out test set — because the same model can look strong
overall while quietly failing on one domain, and that's exactly the kind
of gap this project is built to surface.

## 9. Limitations and honest caveats

- TF-IDF cannot capture negation scope or sarcasm reliably ("not bad at
  all" or "yeah, *great* service" are known failure modes for bag-of-
  n-grams models). A transformer-based model (e.g. fine-tuned
  DistilBERT) would likely close some of this gap at the cost of
  training time and interpretability — noted here as a natural next
  step, not attempted in this iteration.
- The Amazon-to-neutral mapping (3 stars → neutral) is a simplification;
  3-star reviews are sometimes mildly positive or mildly negative rather
  than genuinely neutral in tone.
- Financial PhraseBank's "AllAgree" subset (highest annotator agreement)
  is smaller than the full dataset; this trades data volume for label
  quality, which is the right tradeoff for a small, high-signal domain
  used mainly for the cross-domain comparison.
