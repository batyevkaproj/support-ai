import sounddevice as sd
import numpy as np

DEVICE_ID = 67   # Voicemeeter Out B1 (48000 Hz)

def callback(indata, frames, time, status):
    vol = np.sqrt(np.mean(indata**2))
    print(f"VOL: {vol:.6f}")

print("Listening Voicemeeter B1...")

with sd.InputStream(
    device=DEVICE_ID,
    channels=2,
    samplerate=48000,
    blocksize=1024,
    dtype="float32",
    callback=callback
):
    input("Press Enter to stop\n")
