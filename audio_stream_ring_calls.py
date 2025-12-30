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
DEVICE_ID = 67                 # Voicemeeter Out B1
INPUT_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 512

# ring detection
RING_FREQ_MIN = 400
RING_FREQ_MAX = 450
RING_ENERGY_RATIO = 4.0        # наскільки гудок сильніший за середній спектр
RING_CONFIRM_FRAMES = 5        # скільки блоків поспіль

# call logic
CALL_END_SILENCE = 3.0         # сек тиші = кінець дзвінка
VOICE_THRESHOLD = 0.004

# Whisper
MODEL_SIZE = "large-v3"
LANG = "uk"
PROMPT = "Це телефонна розмова українською мовою."
# =========================================

os.makedirs("calls", exist_ok=True)

# ---------- GPU ----------
cuda_ok = torch.cuda.is_available()
device = "cuda" if cuda_ok else "cpu"
compute = "float16" if cuda_ok else "int8"

print("CUDA:", cuda_ok)
if cuda_ok:
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading Whisper...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute)
print("Whisper loaded")

audio_q = queue.Queue(maxsize=300)

# states
IDLE = 0
RINGING = 1
IN_CALL = 2
state = IDLE

ring_counter = 0
last_voice_time = None
call_audio = []

# -------------------------------------------------

def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def detect_ring_tone(audio_mono, sr):
    spectrum = np.abs(np.fft.rfft(audio_mono))
    freqs = np.fft.rfftfreq(len(audio_mono), 1 / sr)

    band = (freqs > RING_FREQ_MIN) & (freqs < RING_FREQ_MAX)
    if not np.any(band):
        return False

    band_energy = spectrum[band].mean()
    total_energy = spectrum.mean() + 1e-9

    return band_energy > total_energy * RING_ENERGY_RATIO


def callback(indata, frames, time_info, status):
    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        pass


print("🎧 Listening Voicemeeter B1 (ring-based)... Ctrl+C to stop")

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
                block = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            mono = block.mean(axis=1)
            vol = rms(block)
            now = time.time()

            ring = detect_ring_tone(mono, INPUT_RATE)

            # ---------- STATE MACHINE ----------

            if state == IDLE:
                if ring:
                    ring_counter += 1
                    if ring_counter >= RING_CONFIRM_FRAMES:
                        print("🔔 RING DETECTED → CALL START")
                        state = RINGING
                        call_audio = []
                        ring_counter = 0
                else:
                    ring_counter = 0

            elif state == RINGING:
                if not ring and vol > VOICE_THRESHOLD:
                    print("🟢 ANSWER → IN CALL")
                    state = IN_CALL
                    last_voice_time = now
                # ще пишемо гудок (інколи корисно)
                call_audio.append(block)

            elif state == IN_CALL:
                call_audio.append(block)

                if vol > VOICE_THRESHOLD:
                    last_voice_time = now

                if last_voice_time and (now - last_voice_time) > CALL_END_SILENCE:
                    print("🔴 CALL END → TRANSCRIBE")
                    state = IDLE

                    audio_np = np.concatenate(call_audio, axis=0)
                    mono = audio_np.mean(axis=1)

                    # normalize
                    peak = np.max(np.abs(mono)) + 1e-9
                    mono = mono / peak * 0.9

                    audio_16k = resampy.resample(mono, INPUT_RATE, TARGET_RATE)

                    segments, _ = model.transcribe(
                        audio_16k,
                        language=LANG,
                        initial_prompt=PROMPT,
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=True
                    )

                    text = " ".join(s.text.strip() for s in segments).strip()

                    fname = f"calls/call_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(text)

                    print(f"📝 SAVED {fname}")
                    print("TEXT:", text if text else "[EMPTY]")

                    call_audio = []

except KeyboardInterrupt:
    print("\n🛑 Stopped")
    sys.exit(0)
