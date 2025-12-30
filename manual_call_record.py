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
DEVICE_ID = 67              # Voicemeeter Output (B)
INPUT_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 1024

MODEL_SIZE = "large-v3"     # можно заменить на "medium"
LANG = "uk"

MIN_RMS = 0.01              # порог речи
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
    print("\n🛑 Recording stopped")

# ---------- POST PROCESS ----------
if not recorded_audio:
    print("❌ No audio captured")
    sys.exit(0)

print("🔧 Processing audio...")

audio_np = np.concatenate(recorded_audio, axis=0)

# ================= SPLIT CHANNELS =================
operator = audio_np[:, 0]   # LEFT
client   = audio_np[:, 1]   # RIGHT


def transcribe_channel(audio, role):
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"{role} RMS: {rms:.4f}")

    if rms < MIN_RMS:
        print(f"❌ {role}: no speech detected")
        return []

    # normalize
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.9

    # resample
    audio_16k = resampy.resample(audio, INPUT_RATE, TARGET_RATE)

    segments, _ = model.transcribe(
        audio_16k,
        language=LANG,

        beam_size=1,
        best_of=1,
        temperature=0.2,

        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400
        ),

        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
    )

    result = []
    for seg in segments:
        result.append((
            seg.start,
            role,
            seg.text.strip()
        ))

    return result


print("🧠 Transcribing OPERATOR...")
dialog = transcribe_channel(operator, "OPERATOR")

print("🧠 Transcribing CLIENT...")
dialog += transcribe_channel(client, "CLIENT")

# sort by time
dialog.sort(key=lambda x: x[0])

# ---------- SAVE ----------
fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
with open(fname, "w", encoding="utf-8") as f:
    for t, role, text in dialog:
        if text:
            f.write(f"[{t:06.2f}] {role}: {text}\n")

print(f"\n📝 SAVED {fname}")
print("====== DIALOG ======")
for t, role, text in dialog:
    if text:
        print(f"[{t:06.2f}] {role}: {text}")
print("====================")
