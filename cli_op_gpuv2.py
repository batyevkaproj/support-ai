# -*- coding: utf-8 -*-

import os
import sys
import time
from datetime import datetime
import threading

import numpy as np
import sounddevice as sd
import resampy
import torch
from faster_whisper import WhisperModel

# ===================== CONFIG =====================
OPERATOR_DEVICE = 64   # Voicemeeter Out B1
CLIENT_DEVICE   = 65   # Voicemeeter Out B2

INPUT_RATE  = 48000
TARGET_RATE = 16000
BLOCKSIZE   = 1024

MODEL_SIZE = "large-v3"
LANG = "uk"

MIN_RMS = 0.008
NORM_TARGET = 0.9

VAD_MIN_SIL_MS = 900
# ==================================================

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("calls", exist_ok=True)

# ===================== UTILS =======================
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def normalize(x: np.ndarray) -> np.ndarray:
    if rms(x) < MIN_RMS:
        return x
    peak = np.max(np.abs(x)) + 1e-9
    return (x / peak) * NORM_TARGET

def resample_16k(x: np.ndarray) -> np.ndarray:
    return resampy.resample(x, INPUT_RATE, TARGET_RATE)

# ==================================================

print("CUDA:", torch.cuda.is_available())

model = WhisperModel(
    MODEL_SIZE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)

print("\n▶ Натисни ENTER, щоб ПОЧАТИ запис")
input()

# ===================== RECORD ======================
recording = True
op_chunks = []
cl_chunks = []

def make_callback(store):
    def cb(indata, frames, time_info, status):
        store.append(indata.copy())
    return cb

print("🎧 Recording... Натисни ENTER для зупинки\n")

def wait_for_stop():
    global recording
    input()
    recording = False

stop_thread = threading.Thread(target=wait_for_stop)
stop_thread.start()

with sd.InputStream(
    device=OPERATOR_DEVICE,
    channels=1,
    samplerate=INPUT_RATE,
    blocksize=BLOCKSIZE,
    dtype="float32",
    callback=make_callback(op_chunks),
), sd.InputStream(
    device=CLIENT_DEVICE,
    channels=1,
    samplerate=INPUT_RATE,
    blocksize=BLOCKSIZE,
    dtype="float32",
    callback=make_callback(cl_chunks),
):
    while recording:
        time.sleep(0.1)

print("\n🛑 Recording stopped\n")

# ===================== PREP ========================
operator = np.concatenate(op_chunks, axis=0).flatten()
client   = np.concatenate(cl_chunks, axis=0).flatten()

# ===================== TRANSCRIBE ==================
def transcribe(audio: np.ndarray, role: str):
    out = []
    if rms(audio) < MIN_RMS:
        return out

    audio = normalize(audio)
    audio16 = resample_16k(audio)

    segments, _ = model.transcribe(
        audio16,
        language=LANG,
        beam_size=1,
        best_of=1,
        temperature=0.2,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SIL_MS),
        condition_on_previous_text=False,
    )

    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        out.append((float(s.start), role, text))
    return out

print("🧠 Transcribing OPERATOR...")
dialog = transcribe(operator, "OPERATOR")

print("🧠 Transcribing CLIENT...")
dialog += transcribe(client, "CLIENT")

dialog.sort(key=lambda x: x[0])

# ===================== OUTPUT ======================
fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

with open(fname, "w", encoding="utf-8", newline="\n") as f:
    f.write("====== DIALOG ======\n")
    print("====== DIALOG ======")

    for t, role, text in dialog:
        line = f"[{t:07.2f}] {role}: {text}"
        print(line)
        f.write(line + "\n")

    f.write("====================\n")
    print("====================")

print(f"\n📝 SAVED {fname}")
