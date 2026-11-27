# PinkFlag web app (FastAPI + Sisi + detector + local voice), for Hugging Face Spaces or any Docker host.
FROM python:3.12-slim

# Spaces run as a non-root user with uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir -e ".[voice]"

COPY --chown=user configs ./configs
COPY --chown=user web ./web
# trained detector + Sisi's voice (not in git: uploaded with the Space, see tools/deploy_space.py)
COPY --chown=user models ./models
RUN mkdir -p /app/data && chown -R user /app/data

USER user
ENV HOME=/home/user HOST=0.0.0.0 PORT=7860 PINKFLAG_DAILY_LIMIT=40 \
    PINKFLAG_WHISPER=base HF_HUB_DISABLE_SYMLINKS_WARNING=1
EXPOSE 7860
# GEMINI_API_KEY comes from the Space's secrets, never from the image
CMD ["python", "-m", "pinkflag.app"]
