import sounddevice as sd

print("=== LOOPBACK DEVICES ===")
for i, d in enumerate(sd.query_devices()):
    name = d["name"].lower()
    if "loopback" in name or "wasapi" in name:
        print(i, d)
