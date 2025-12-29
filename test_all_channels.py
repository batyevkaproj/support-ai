import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 48000
DURATION = 0.5   # секунд на устройство

def rms(data):
    return float(np.sqrt(np.mean(data**2)))

devices = sd.query_devices()

print("\n=== AUDIO INPUT SCAN ===\n")

for i, d in enumerate(devices):
    if d['max_input_channels'] < 1:
        continue

    print(f"\n--- Device {i}: {d['name']} ---")

    try:
        with sd.InputStream(
            device=i,
            channels=1,
            samplerate=int(d['default_samplerate']),
            blocksize=1024,
            dtype="float32"
        ) as stream:
            data = sd.rec(
                int(SAMPLE_RATE * DURATION),
                samplerate=int(d['default_samplerate']),
                channels=1,
                dtype="float32",
                device=i
            )
            sd.wait()
            vol = rms(data)
            print(f"RMS VOLUME: {vol:.8f}")

    except Exception as e:
        print("ERROR:", e)

print("\n=== SCAN FINISHED ===")
