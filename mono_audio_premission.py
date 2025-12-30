import sounddevice as sd

for i, d in enumerate(sd.query_devices()):
    if "Voicemeeter" in d["name"]:
        print(i, d["name"], "in:", d["max_input_channels"])
