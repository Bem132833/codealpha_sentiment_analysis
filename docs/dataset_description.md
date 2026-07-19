# Dataset Description

This project deliberately uses **three datasets from three different text
domains**, matching the task brief's scope ("Reviews/social media/news")
rather than picking a single easy source. This is what makes the
cross-domain generalization analysis in this project possible.

## 1. Amazon Product Reviews (domain: `amazon_reviews`)

- **Source:** Kaggle — "Amazon Reviews for Sentiment Analysis"
  (originally derived from the Stanford Network Analysis Project's
  Amazon review corpus)
- **Domain type:** Long-form, opinionated consumer text
- **Original size:** Millions of rows upstream; this project samples
  down to `config.AMAZON_SAMPLE_SIZE` (default 20,000) for tractability
- **Original labels:** Star ratings, 1–5
- **Label mapping used in this project:**
  | Stars | Mapped label |
  |---|---|
  | 1–2 | negative |
  | 3 | neutral |
  | 4–5 | positive |
- **Expected columns:** a text column (`reviewText`, `text`, or
  `review_text`) and a rating column (`overall`, `rating`, `score`, or
  `stars`)
- **Download into:** `data/raw/amazon_reviews/amazon_reviews.csv`

## 2. Twitter US Airline Sentiment (domain: `twitter_airline`)

- **Source:** Kaggle — "Twitter US Airline Sentiment" (Crowdflower/Figure
  Eight, originally collected February 2015)
- **Domain type:** Short, informal, often sarcastic social media text
- **Size:** 14,640 tweets
- **Original labels:** Already three-class (`positive`, `negative`,
  `neutral`) — no mapping needed
- **Expected columns:** `text` and `airline_sentiment`
- **Download into:** `data/raw/twitter_airline/twitter_airline_sentiment.csv`

## 3. Financial PhraseBank (domain: `financial_phrasebank`)

- **Source:** Kaggle / HuggingFace — `financial_phrasebank` (Malo et al.,
  2014, "Good Debt or Bad Debt: Detecting Semantic Orientations in
  Economic Texts")
- **Domain type:** Formal, understatement-heavy financial news sentences,
  labeled by finance professionals — the hardest domain for a general
  sentiment model, and the domain with the highest proportion of neutral
  examples
- **Size:** ~5,000 sentences (this project uses the subset where
  annotator agreement was highest, commonly distributed as
  `Sentences_AllAgree.txt` / a pre-converted CSV)
- **Original labels:** Already three-class (`positive`, `negative`,
  `neutral`)
- **Expected columns:** `sentence` (or `text`) and `label`
- **Download into:** `data/raw/financial_phrasebank/financial_phrasebank.csv`

## Why three domains instead of one

Most public sentiment-analysis portfolio projects train and test on a
single dataset, which tells you nothing about whether the model actually
learned "sentiment" versus just learned the surface patterns of one
narrow style of text. Using three domains with genuinely different
properties — review length, formality, sarcasm, class balance — lets
this project measure and report that generalization gap directly (see
`cross_domain_eval.py` and `docs/methodology.md`), rather than asserting
it.

## Licensing note

All three datasets are publicly available for research and educational
use under their respective Kaggle/original dataset licenses. No raw data
files are committed to this repository (see `.gitignore`) — download them
from the sources above and place them in the paths listed. If a raw file
is missing, `src/data_loader.py` automatically falls back to generating a
small synthetic sample per domain so the pipeline remains runnable
end-to-end for development and testing.
