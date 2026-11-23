"""Sisi's ears and voice, 100% local and free (no audio ever leaves your computer).

  ears:   faster-whisper (OpenAI's Whisper, optimized). Speech -> text, detects the language.
          Runs on the GPU when it can, otherwise on the CPU.
  voice:  Piper. Text -> natural speech, a small neural voice model (~60 MB).

Both models load lazily (on the first call) and stay in memory afterwards.
"""
import io
import os
import re
import wave
from functools import lru_cache
from pathlib import Path

VOICE_PATH = Path("models/voices/en_US-amy-medium.onnx")
WHISPER_SIZE = os.environ.get("PINKFLAG_WHISPER", "small")  # tiny/base/small/medium

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def _add_cuda_dlls() -> None:
    """On Windows, Whisper's GPU backend needs the CUDA DLLs that ship with PyTorch."""
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if hasattr(os, "add_dll_directory") and lib.exists():
            os.add_dll_directory(str(lib))
    except ImportError:
        pass


@lru_cache(maxsize=1)
def whisper_model():
    from faster_whisper import WhisperModel
    _add_cuda_dlls()
    try:
        return WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16"), "gpu"
    except Exception:  # noqa: BLE001 - no usable GPU: fall back to the CPU
        return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8"), "cpu"


@lru_cache(maxsize=1)
def piper_voice():
    from piper import PiperVoice
    if not VOICE_PATH.exists():
        raise FileNotFoundError(
            f"{VOICE_PATH} missing: run python -m piper.download_voices "
            f"--download-dir models/voices en_US-amy-medium")
    return PiperVoice.load(str(VOICE_PATH))


def transcribe(audio: bytes) -> dict:
    """Any audio the browser records (webm/ogg/wav) -> {"text", "language"}."""
    model, _ = whisper_model()
    # the voice-activity filter skips silence, but can also drop quiet voices: if it
    # throws everything away, listen again without it
    for vad in (True, False):
        segments, info = model.transcribe(io.BytesIO(audio), vad_filter=vad, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            break
    return {"text": text, "language": info.language}


def speakable(text: str) -> str:
    """What Sisi says out loud: no emojis, no markdown, no quote-card clutter."""
    text = EMOJI.sub("", text)
    text = re.sub(r"[*_#`>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text: str) -> bytes:
    """Text -> WAV bytes in Sisi's voice."""
    voice = piper_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(speakable(text) or "mm-hm", wav)
    return buf.getvalue()
