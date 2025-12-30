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
DEVICE_ID = 67              # Voicemeeter Output (B) device id у sounddevice
INPUT_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 1024

MODEL_SIZE = "large-v3"     # якщо буде важко/нестабільно: "medium"
LANG = "uk"

# Розумні пороги
MIN_RMS = 0.008             # нижче — вважаємо тишею/шумом
NORM_TARGET = 0.90

# Chunking (критично для стабільності)
CHUNK_SECONDS = 12.0        # 10–15 сек найкраще
CHUNK_OVERLAP = 1.0         # перекриття, щоб не різало слова

# VAD tuning
VAD_MIN_SIL_MS = 900        # збільшили (було 400) -> менше "по слову"
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

# Split channels: LEFT=operator, RIGHT=client
operator = audio_np[:, 0].astype(np.float32)
client   = audio_np[:, 1].astype(np.float32)

def rms_level(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def normalize_if_needed(x: np.ndarray) -> np.ndarray:
    # Не нормалізуємо "порожнечу" — це викликає галюцинації
    r = rms_level(x)
    if r < MIN_RMS:
        return x
    peak = float(np.max(np.abs(x)) + 1e-9)
    return (x / peak) * NORM_TARGET

def resample_16k(x: np.ndarray) -> np.ndarray:
    if INPUT_RATE == TARGET_RATE:
        return x
    return resampy.resample(x, INPUT_RATE, TARGET_RATE)

def chunk_indices(n_samples: int, sr: int, chunk_sec: float, overlap_sec: float):
    chunk = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    step = max(1, chunk - overlap)
    i = 0
    while i < n_samples:
        j = min(n_samples, i + chunk)
        yield i, j
        if j == n_samples:
            break
        i += step

def transcribe_chunk(audio_16k: np.ndarray):
    # faster-whisper приймає np.float32 16k mono
    segments, _ = model.transcribe(
        audio_16k,
        language=LANG,

        beam_size=1,
        best_of=1,
        temperature=0.2,

        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=VAD_MIN_SIL_MS
        ),

        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
    )
    return segments

def transcribe_channel_chunked(raw_audio: np.ndarray, role: str):
    dialog = []
    total_samples = raw_audio.shape[0]
    total_sec = total_samples / INPUT_RATE
    print(f"🧾 {role}: {total_sec:.1f}s, RMS={rms_level(raw_audio):.4f}")

    for i, j in chunk_indices(total_samples, INPUT_RATE, CHUNK_SECONDS, CHUNK_OVERLAP):
        chunk_raw = raw_audio[i:j]
        r = rms_level(chunk_raw)
        if r < MIN_RMS:
            continue

        chunk_raw = normalize_if_needed(chunk_raw)
        chunk_16k = resample_16k(chunk_raw)

        # offset у секундах від початку всього запису
        base_offset = i / INPUT_RATE

        segments = transcribe_chunk(chunk_16k)
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            # seg.start/seg.end у межах чанка -> додаємо base_offset
            start = float(seg.start) + base_offset
            end = float(seg.end) + base_offset
            dialog.append((start, end, role, text))

    return dialog

# 1) Транскрибуємо окремо канали, але ПО ЧАНКАХ + глобальний час
print("🧠 Transcribing OPERATOR (LEFT)...")
dialog = transcribe_channel_chunked(operator, "OPERATOR")

print("🧠 Transcribing CLIENT (RIGHT)...")
dialog += transcribe_channel_chunked(client, "CLIENT")

# 2) Сортуємо по глобальному часу
dialog.sort(key=lambda x: x[0])

# 3) Дедуп / анти-ехо
def normalize_text(t: str) -> str:
    # дуже просте нормалізування
    t = t.lower().strip()
    t = " ".join(t.split())
    return t

def postprocess(dialog_items):
    cleaned = []
    last_norm = ""
    last_time = -999.0

    for start, end, role, text in dialog_items:
        nt = normalize_text(text)

        # прибираємо повні дублікати підряд
        if nt == last_norm and (start - last_time) < 1.5:
            continue

        cleaned.append((start, end, role, text))
        last_norm = nt
        last_time = start

    # анти-ехо: якщо однакова фраза з’явилась в обох ролях майже одночасно — лишаємо одну (гучнішу ми не знаємо, тож лишимо першу)
    final = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        if i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            if abs(nxt[0] - cur[0]) < 0.6 and normalize_text(nxt[3]) == normalize_text(cur[3]) and nxt[2] != cur[2]:
                final.append(cur)   # залишаємо першу
                i += 2
                continue
        final.append(cur)
        i += 1

    return final

dialog = postprocess(dialog)

# 4) Збереження
fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write("====== DIALOG ======\n")
    for start, end, role, text in dialog:
        f.write(f"[{start:07.2f}] {role}: {text}\n")
    f.write("====================\n")

print(f"\n📝 SAVED {fname}")
print("====== DIALOG ======")
for start, end, role, text in dialog:
    print(f"[{start:07.2f}] {role}: {text}")
print("====================")
