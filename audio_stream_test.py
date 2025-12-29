import sounddevice as sd
import numpy as np

DEVICE_ID = 13  # <-- твой CABLE Output

dev = sd.query_devices(DEVICE_ID)
print("Using device:", dev)

def callback(indata, frames, time, status):
    if status:
        print(status)
    volume = np.linalg.norm(indata)
    if volume > 0.0005:
        print("Volume:", round(volume, 4))

print("Listening CABLE Output...")

with sd.InputStream(
    device=DEVICE_ID,
    channels=1,                 # важно: mono
    samplerate=48000,            # как в VB-Cable
    blocksize=512,
    dtype="float32",
    callback=callback
):
    input("Press Enter to stop\n")
