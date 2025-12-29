import sounddevice as sd
import numpy as np
import queue
import time
import os
import sys
from datetime import datetime
from faster_whisper import WhisperModel

# ================== CONFIG ==================
DEVICE_ID = 67                 # Voicemeeter Out B1 (48000 Hz)
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCKSIZE = 1024

VOLUME_THRESHOLD = 0.004       # детект речи
SILENCE_TIMEOUT = 1.2          # секунд тишины = конец фразы

MODEL_SIZE = "small"           # можно потом medium
LANG = None                    # авто
# ============================================

os.makedirs("calls", exist_ok=True)

print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)
print("Model loaded")

audio_queue = queue.Queue(maxsize=50)

current_audio = []
last_voice_time = None
call_active = False
running = True


def rms(data):
    return float(np.sqrt(np.mean(data ** 2)))


def callback(indata, frames, time_info, status):
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


print("🎧 Listening Zoiper via Voicemeeter B1 (Ctrl+C to stop)")

try:
    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=callback
    ):
        while running:
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            vol = rms(data)
            now = time.time()

            if vol > VOLUME_THRESHOLD:
                if not call_active:
                    print("📞 Speech detected")
                    call_active = True
                    current_audio = []
                last_voice_time = now
                current_audio.append(data)

            if call_active and last_voice_time and (now - last_voice_time) > SILENCE_TIMEOUT:
                print("📴 Silence → transcribing")
                call_active = False

                audio_np = np.concatenate(current_audio, axis=0).flatten()

                segments, _ = model.transcribe(
                    audio_np,
                    language=LANG,
                    vad_filter=True
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

    if current_audio:
        print("⏳ Transcribing last audio...")
        audio_np = np.concatenate(current_audio, axis=0).flatten()
        segments, _ = model.transcribe(audio_np)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print("FINAL TEXT:", text)

    sys.exit(0)
