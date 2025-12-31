# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import resampy
import torch
from faster_whisper import WhisperModel

# ================= CONFIG =================
OPERATOR_DEVICE = 64   # Voicemeeter Out B1
CLIENT_DEVICE   = 65   # Voicemeeter Out B2

INPUT_RATE  = 48000
TARGET_RATE = 16000
BLOCKSIZE   = 1024

MODEL_SIZE = "large-v3"
LANG = "uk"

MIN_RMS = 0.008
NORM_TARGET = 0.9

CHUNK_SECONDS = 12.0
CHUNK_OVERLAP = 1.0
VAD_MIN_SIL_MS = 900
# ========================================

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("calls", exist_ok=True)

print("CUDA available:", torch.cuda.is_available())

model = WhisperModel(
    MODEL_SIZE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)

# ================= UTILS =================
def rms(x):
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def normalize(x):
    if rms(x) < MIN_RMS:
        return x
    peak = np.max(np.abs(x)) + 1e-9
    return (x / peak) * NORM_TARGET

def resample_16k(x):
    return resampy.resample(x, INPUT_RATE, TARGET_RATE)

def chunk_indices(n_samples, sr, chunk_sec, overlap_sec):
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
# ========================================

def transcribe_channel(audio, role):
    results = []
    total = audio.shape[0]

    for i, j in chunk_indices(total, INPUT_RATE, CHUNK_SECONDS, CHUNK_OVERLAP):
        chunk = audio[i:j]
        if rms(chunk) < MIN_RMS:
            continue

        chunk = normalize(chunk)
        chunk16 = resample_16k(chunk)
        base_offset = i / INPUT_RATE

        segments, _ = model.transcribe(
            chunk16,
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

        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            start = float(s.start) + base_offset
            results.append((start, role, text))

    return results

# ================= MAIN LOOP =================
print("\n🟢 Система готова")
print("👉 Натисни ENTER — ПОЧАТИ / ЗАВЕРШИТИ розмову")
print("👉 Ctrl+C — ВИЙТИ\n")

while True:
    input("▶ ENTER — почати запис ")

    recording = True
    op_chunks = []
    cl_chunks = []

    def make_callback(store):
        def cb(indata, frames, time_info, status):
            mono = indata[:, 0].copy()  # stereo → mono
            store.append(mono)
        return cb

    def wait_for_stop():
        nonlocal_recording[0] = False

    nonlocal_recording = [True]

    def stop_waiter():
        input("⏹ ENTER — завершити запис ")
        nonlocal_recording[0] = False

    stopper = threading.Thread(target=stop_waiter, daemon=True)
    stopper.start()

    print("🎧 RECORDING...\n")

    with sd.InputStream(
        device=OPERATOR_DEVICE,
        channels=2,
        samplerate=INPUT_RATE,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=make_callback(op_chunks),
    ), sd.InputStream(
        device=CLIENT_DEVICE,
        channels=2,
        samplerate=INPUT_RATE,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=make_callback(cl_chunks),
    ):
        while nonlocal_recording[0]:
            time.sleep(0.1)

    print("\n🛑 Запис завершено. Транскрипція...\n")

    operator = np.concatenate(op_chunks, axis=0)
    client   = np.concatenate(cl_chunks, axis=0)

    dialog = []
    dialog += transcribe_channel(operator, "OPERATOR")
    dialog += transcribe_channel(client, "CLIENT")

    dialog.sort(key=lambda x: x[0])

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

    print(f"\n📝 SAVED {fname}\n")
