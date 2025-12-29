import sounddevice as sd
import numpy as np

DEVICE_ID = 13   # <-- временно, потом проверим
SAMPLE_RATE = 48000

def callback(indata, frames, time, status):
    vol = float(np.sqrt(np.mean(indata**2)))
    print(f"VOLUME: {vol:.8f}")

print("Listening raw volume...")

with sd.InputStream(
    device=DEVICE_ID,
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=1024,
    dtype="float32",
    callback=callback
):
    input("Speak in Zoiper, press Enter to stop\n")
