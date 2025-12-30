import os
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import resampy
import torch
from faster_whisper import WhisperModel

# ================= CONFIG =================
DEVICE_ID = 67            # Voicemeeter Out B1
INPUT_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 1024

MODEL_SIZE = "large-v3"   # RTX 4080 тягне
LANG = "uk"
PROMPT = "Це телефонна розмова українською мовою."
MIN_RMS = 0.01            # ❗ мін. рівень мовлення
# =========================================

os.makedirs("calls", exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)
print("Model loaded")

recorded_audio = []


def callback(indata, frames, time_info, status):
    recorded_audio.append(indata.copy())


print("\n🎧 RECORDING STARTED")
print("👉 Запис іде. Натисни Ctrl+C коли дзвінок завершиться.\n")

try:
    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=INPUT_RATE,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=callback
    ):
        while True:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Recording stopped by user")

# ---------- POST PROCESS ----------
if not recorded_audio:
    print("❌ No audio captured")
    sys.exit(0)

print("🔧 Processing audio...")

audio_np = np.concatenate(recorded_audio, axis=0)

# stereo → mono
mono = audio_np.mean(axis=1)

# RMS check (❗ КРИТИЧНО)
rms = np.sqrt(np.mean(mono ** 2))
print(f"🔊 RMS level: {rms:.4f}")

if rms < MIN_RMS:
    print("❌ No speech detected (too quiet), skipping transcription")
    sys.exit(0)

# normalize ONLY if speech exists
peak = np.max(np.abs(mono)) + 1e-9
mono = mono / peak * 0.9

# resample to 16k
audio_16k = resampy.resample(mono, INPUT_RATE, TARGET_RATE)

print("🧠 Transcribing...")

segments, info = model.transcribe(
    audio_16k,
    language=LANG,
    initial_prompt=PROMPT,

    beam_size=1,
    best_of=1,
    temperature=0.2,

    vad_filter=True,                     # ❗ ВКЛЮЧИТИ
    vad_parameters=dict(
        min_silence_duration_ms=500
    ),

    condition_on_previous_text=False,    # ❗ КЛЮЧОВЕ
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.0,
)

text = " ".join(seg.text.strip() for seg in segments).strip()

fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n📝 SAVED {fname}")
print("====== TEXT ======")
print(text if text else "[EMPTY]")
print("==================")
