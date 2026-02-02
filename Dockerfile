# AI-SUB v6.0 Dockerfile - Library Manager with TMDB Integration
# Intel Battlemage (XPU) Ready
FROM intel/intel-extension-for-pytorch:2.6.10-xpu

USER root

# 1. Install System Dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pciutils \
    python3-dev \
    build-essential \
    mkvtoolnix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
RUN pip install --no-cache-dir \
    tiktoken \
    tqdm \
    more-itertools \
    numpy \
    numba \
    scipy \
    setuptools \
    ffsubsync \
    requests

# 3. Install FastAPI and uvicorn for API server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles

# 4. Install librosa for audio analysis
RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile \
    audioread \
    resampy

# 5. Install Whisper packages WITHOUT breaking drivers (--no-deps is critical)
RUN pip install --no-cache-dir --no-deps \
    openai-whisper \
    stable-ts

# 6. Create folder structure
WORKDIR /app
RUN mkdir -p /data /output /media

# 7. Copy script
COPY script.py /app/script.py

# 8. Environment variables
ENV WHISPER_MODEL="large-v2" \
    WEBHOOKPORT="9000" \
    DEBUG="true" \
    # Data and output directories
    DATA_DIR="/data" \
    OUTPUT_DIR="/output" \
    # TMDB API (required for library management)
    TMDB_API_KEY="" \
    # Library scanning interval (seconds)
    LIBRARY_SCAN_INTERVAL="300" \
    # Muxing options
    MUX_SUBTITLES="true" \
    REPLACE_ORIGINAL="true" \
    DELETE_SRT_AFTER_MUX="true" \
    # VAD (Silero)
    VAD_ENABLED="true" \
    VAD_THRESHOLD="0.35" \
    # stable-ts features
    SUPPRESS_SILENCE="true" \
    SUPPRESS_WORD_TS="true" \
    USE_REGROUP="true" \
    CUSTOM_REGROUP="cm_sl=84_sl=42++++++1" \
    # Model prompt
    USE_MODEL_PROMPT="true" \
    MODEL_PROMPT="Hello, welcome to my video." \
    # Performance
    COMPUTE_TYPE="auto" \
    WHISPER_THREADS="4" \
    CONCURRENT_TRANSCRIPTIONS="1" \
    CLEAR_VRAM_ON_COMPLETE="true" \
    # Audio preprocessing
    USE_DEMUCS="false" \
    ONLY_VOICE_FREQ="false" \
    # Output formatting
    MAX_CHARS_PER_LINE="42" \
    MIN_SEGMENT_DURATION="0.5" \
    # Language detection
    LANG_SAMPLE_SECONDS="30" \
    LANG_SAMPLE_OFFSET="60" \
    ALLOW_UNKNOWN_AS_ENGLISH="true" \
    # Subtitle sync - DISABLED by default (can cause issues on some files)
    SYNC_SUBTITLES="false"

# 9. Expose API port
EXPOSE 9000

# 10. Entrypoint
CMD ["python", "script.py"]
