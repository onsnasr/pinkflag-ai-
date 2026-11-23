# 🛣️ Roadmap

Each version has one clear goal and one thing you learn from building it.

## v0.1 Baseline ✅
- Taxonomy of 8 labels with written definitions
- Hand-written seed dataset (Gen Z slang included)
- TF-IDF + Logistic Regression, cross-validated metrics
- Tests + GitHub Actions CI
- **You learn:** text → numbers, why a simple baseline comes first, how cross-validation keeps you honest

## v0.2 Real data ✅
- Responsible web scraping (Beautiful Soup, robots.txt, rate limit, cache) + weak labeling
- **MentalManip** (ACL 2024) converted, label mapping documented
- Tunisian Derja: TSAC + TUNIZI, YouTube Data API, Gemini translation (data augmentation)
- One merged dataset with group-aware train / val / test splits (no leakage)
- **You learn:** data is the model; label noise; licenses; leakage

## v0.3 Sisi, the agent ✅ (terminal)
- Gemini + function calling: `analyze_messages` (our model), `explain_flag`, `remember`, `safety_resources`
- Persona: warm big sister, Gen Z voice, replies in the user's language, never diagnoses people
- Memory: conversation + long-term notes in SQLite
- Retries + fallback models when the API is busy
- **You learn:** tool use, prompt design, grounding an LLM in your own model, agent memory

## v0.4 Transformer + experiment tracking ⏳
- Fine-tune **XLM-RoBERTa** (multilingual) on the merged dataset; compare with the baseline
- Plug the better model into Sisi's `analyze_messages`
- **MLflow** for every run; **DVC** pipeline (`dvc repro` from scraping to training)
- Later: multi-label, probability calibration, whole-conversation classification
- **You learn:** transfer learning, deep learning in PyTorch, experiment tracking

## v0.5 Chat app 💖
- Design: `web/mockup/chat.html`: the Sisi conversation takes the screen, "receipts" panel on the side, one case file per guy
- **FastAPI** backend (`/chat` with streaming, `/upload` for screenshots), Docker
- Frontend built from the mockup; Sisi's avatar (brown hair, glasses) with a few expressions
- **You learn:** serving ML, APIs, frontend

## v0.6 Screenshots 📸
- Gemini vision reads the screenshot: messages + who said what (left/right bubbles)
- Optional comparison with OCR (PaddleOCR / Tesseract)
- **You learn:** multimodal models, evaluating extraction accuracy

## v0.7 Voice 🎙️🔊
- **Voice notes in:** mic button in the chat box; the recorded audio goes to Gemini, which understands
  English / French / Derja mixes (fallback: the browser's built-in speech recognition for English)
- **Sisi speaks back:** text-to-speech with a warm big-sister voice, ▶️ button on her replies
- Optional: local **Whisper** on the GPU for private transcription
- **You learn:** speech-to-text, text-to-speech, audio in web apps

## v1.0 Ship it 🚀
- **Free public hosting** on Hugging Face Spaces (one link any girl can open on her phone)
  - fair-use limit per person (Gemini free quota), no chats stored on the server by default,
    privacy note, "Sisi isn't a therapist" note + help resources
- **Demo video (~60s):** script + storyboard → automated screen recording of the app →
  Ons's own voiceover → combined with ffmpeg (captions + music optional)
- Feedback loop: "was Sisi right? 👍👎" → new labeled data → retrain
- Model card: what it's trained on, where it fails, who shouldn't rely on it
- Privacy: chats stay private, one-tap "delete everything"
- **You learn:** the MLOps loop: data → train → deploy → monitor → retrain
