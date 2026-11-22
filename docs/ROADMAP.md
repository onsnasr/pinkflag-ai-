# 🛣️ Roadmap

Each version has one clear goal and one thing you learn from building it.

## v0.1 Baseline ✅
- Taxonomy of 7 labels with written definitions
- Seed dataset (185 hand-written messages, Gen Z slang included)
- TF-IDF + Logistic Regression, cross-validated metrics
- Tests + GitHub Actions CI
- **You learn:** text → numbers, why a simple baseline comes first, how cross-validation keeps you honest

## v0.2 Real data
- Ingest **MentalManip** (ACL 2024, 4k annotated manipulative dialogues) and map its labels to ours
- Generate synthetic **Gen Z conversations** (multi-turn, not single texts) from patterns written down from TikTok discussions; human review every row
- Write a **labeling guide** with examples + edge cases; label a sample twice and measure agreement (Cohen's kappa)
- Version data with **DVC**
- **You learn:** data is the model; label noise; why test data must come from a different source

## v0.3 Better models + experiment tracking
- Sentence-embedding model + classifier
- Fine-tuned **DeBERTa-v3-small**, multi-label (a message can be guilt-tripping *and* controlling)
- **Probability calibration**, so "80%" actually means right 80% of the time
- **MLflow** for every run: params, metrics, artifacts; compare models side by side
- Compare against an LLM with no training (zero-shot)
- **You learn:** transfer learning, multi-label, calibration, experiment tracking

## v0.4 Screenshot reader
- Option A: OCR (PaddleOCR / Tesseract) + bubble position → who's speaking
- Option B: vision LLM reads the screenshot directly
- Build a small labeled set of screenshots and measure both
- **You learn:** CV pipelines, evaluating extraction accuracy

## v0.5 PinkFlag agent
- LLM agent with tools: `classify_messages`, `explain_flags`, `conversation_summary`
- Persona: warm big-sister energy, Gen Z voice, always shows receipts (quotes the exact messages)
- Guardrails: never diagnoses people, points to help resources when something is serious
- **You learn:** tool use, prompt design, grounding answers in model outputs

## v0.6 Pink web app
- **FastAPI** backend (`/analyze` text + image), Docker
- **Next.js** frontend: upload, verdict cards, highlighted receipts
- **You learn:** serving ML, APIs, frontend

## v1.0 Ship it
- Deploy (Hugging Face Spaces / Render / Vercel)
- Monitoring: prediction drift, user feedback ("was PinkFlag right?") → new labeled data
- Model card: what it's trained on, where it fails, who shouldn't rely on it
- **You learn:** the MLOps loop: data → train → deploy → monitor → retrain
