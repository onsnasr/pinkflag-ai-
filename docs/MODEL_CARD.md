# Model card: PinkFlag red-flag classifier

## What it does
Classifies a single message from a dating conversation into one of 8 labels: `green_flag`,
`love_bombing`, `gaslighting`, `controlling`, `guilt_tripping`, `disrespect`, `intimidation`,
`breadcrumbing`. Inside the app, the prediction is a tool called by the Sisi agent; the agent
explains it, the model decides it.

## Models compared
| Model | Type | Parameters | Trained by us |
|---|---|---|---|
| TF-IDF (word 1-2 + char 2-5 grams) + Logistic Regression | classical ML | sparse linear | from scratch |
| BiLSTM (2 layers, bidirectional, max-pool) | deep learning, no pre-training | 1.2M | from scratch |
| XLM-RoBERTa base + classification head | deep learning, transfer learning | 278M (57M trainable) | fine-tuned |

## Training data
Merged from 4 sources (see `docs/LABEL_MAPPING.md`, `src/pinkflag/build_dataset.py`):
hand-written English / Gen Z messages, scraped advice-article quotes (weak labels),
MentalManip (ACL 2024, movie-script dialogues, CC BY-SA 4.0) and Gemini translations into
Tunisian Derja. **Final models train on trusted labels only (591 rows).**

Label-noise investigation: an independent LLM annotator (Gemini) agreed with our loose
MentalManip mappings only **33% of the time (Cohen's kappa 0.15)**. Adding the rows where both
agreed did not improve results (ablation), so they are excluded.

## Evaluation
Group-aware stratified split (no leakage between a message and its translations).
Reported on the **clean test set** (n = 114, trusted labels only) with 95% bootstrap intervals:

| Model | Accuracy | Macro-F1 | Red flag vs green |
|---|---|---|---|
| TF-IDF + LogReg | 61% (52-69) | 0.56 (0.43-0.67) | 97% (94-100) |
| BiLSTM from scratch | 39% (30-47) | 0.35 (0.26-0.42) | 93% (89-97) |
| XLM-RoBERTa fine-tuned | 63% (54-72) | 0.58 (0.46-0.67) | 94% (89-97) |

**Reading:** with ~600 labeled examples, a network trained from scratch cannot learn language
(BiLSTM clearly worst, intervals don't overlap). Pre-training recovers it: XLM-RoBERTa matches or
beats the classical baseline, though their intervals overlap, so the difference is not
statistically meaningful at this test size. The binary question (red flag or not) is solved
much better than the 8-way type.

Reproduce: `python -m pinkflag.evaluate` (after training each model).

## Limitations
- Small, partly synthetic dataset; the clean test set is small, hence wide intervals.
- Single messages, not whole conversations: context changes meaning ("where are you?").
- Movie dialogue and hand-written examples are not real texting; Derja coverage is weak.
- Labels are subjective; no inter-annotator agreement on real messages yet.

## Intended use and ethics
A reflection aid, not a judgment of anyone's character, and not a substitute for professional
help. Scores describe the messages shared, never the person. No private social-media data was
scraped. When messages suggest danger, the agent points to real-world help.
