import os
import sys
import time
import queue
from datetime import datetime

import numpy as np
import sounddevice as sd
import resampy
import torch
from faster_whisper import WhisperModel

# ================= CONFIG =================
DEVICE_ID = 67                 # Voicemeeter Out B1 (у тебя были цифры)
INPUT_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 512                # меньше = меньше задержка

# детект речи
VOLUME_THRESHOLD = 0.0035      # под твои уровни 0.01..0.17
SILENCE_TIMEOUT = 0.55         # меньше = быстрее реакция
MIN_AUDIO_SECONDS = 0.5        # не режем слишком коротко

# Whisper
MODEL_SIZE = "large-v3"          # GPU тянет; если надо быстрее -> "small"
LANG = "uk"
PROMPT = "Це телефонна розмова українською мовою."

# Whisper speed/quality knobs
BEAM_SIZE = 1                  # быстрее
VAD_FILTER = False             # мы сами режем по тишине
CONDITION_ON_PREV = False      # быстрее/стабильнее для кусочков
# =========================================

os.makedirs("calls", exist_ok=True)

# ---------- GPU check ----------
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
if cuda_ok:
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

device = "cuda" if cuda_ok else "cpu"
compute_type = "float16" if cuda_ok else "int8"

print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device=device,
    compute_type=compute_type
)
print(f"Model loaded ({MODEL_SIZE}, device={device}, compute={compute_type})")

audio_queue = queue.Queue(maxsize=200)

current_audio = []
last_voice_time = None
segment_active = False


def rms(x: np.ndarray) -> float:
    # x shape: (frames, channels)
    return float(np.sqrt(np.mean(x ** 2)))


def callback(indata, frames, time_info, status):
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


def prepare_audio_float32(audio_stereo: np.ndarray) -> np.ndarray:
    """
    audio_stereo: float32, shape (N, 2)
    returns: float32 mono 16kHz
    """
    # stereo -> mono
    mono = audio_stereo.mean(axis=1).astype(np.float32)

    # normalize (защита от клиппинга/тихого уровня)
    peak = float(np.max(np.abs(mono)) + 1e-9)
    mono = (mono / peak) * 0.9

    # resample 48k -> 16k
    audio_16k = resampy.resample(mono, INPUT_RATE, TARGET_RATE).astype(np.float32)
    return audio_16k


print("🎧 Listening Voicemeeter B1... (Ctrl+C to stop)")
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
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            vol = rms(data)
            now = time.time()

            # старт речи
            if vol > VOLUME_THRESHOLD:
                if not segment_active:
                    print("📞 Speech detected")
                    segment_active = True
                    current_audio = []
                last_voice_time = now
                current_audio.append(data)

            # конец речи по тишине
            if segment_active and last_voice_time and (now - last_voice_time) > SILENCE_TIMEOUT:
                segment_active = False
                print("📴 Silence → transcribing")

                if not current_audio:
                    continue

                audio_np = np.concatenate(current_audio, axis=0)
                dur_in = len(audio_np) / INPUT_RATE
                if dur_in < MIN_AUDIO_SECONDS:
                    print("⚠️ Too short, skipping")
                    continue

                audio_16k = prepare_audio_float32(audio_np)
                dur_16k = len(audio_16k) / TARGET_RATE
                if dur_16k < MIN_AUDIO_SECONDS:
                    print("⚠️ Too short after resample, skipping")
                    continue

                segments, _ = model.transcribe(
                    audio_16k,
                    language=LANG,
                    initial_prompt=PROMPT,
                    beam_size=BEAM_SIZE,
                    vad_filter=VAD_FILTER,
                    condition_on_previous_text=True,
                )

                text = " ".join(seg.text.strip() for seg in segments).strip()

                if text:
                    fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"📝 SAVED {fname}")
                    print("TEXT:", text)
                else:
                    print("⚠️ No text recognized")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
    sys.exit(0)
