import json, subprocess, sys, os
S = sys.argv[1]
d = json.load(open(f"{S}/artifact-files/db972504-b155-4c32-825c-c256a40b157e/script.json"))
V = {"narrator": ("dmitri", 1.18, 0.96), "future": ("ruslan", 1.12, 0.94), "hero": ("denis", 1.0, 1.06)}
out = []
for sc in d["scenes"]:
    for i, l in enumerate(sc["lines"]):
        v, ls, pitch = V[l["sp"]]
        raw = f"{S}/build/vo/{sc['id']}_{i}_raw.wav"; fin = f"{S}/build/vo/{sc['id']}_{i}.wav"
        subprocess.run(["piper", "-m", f"{S}/voices/ru_RU-{v}-medium.onnx", "-f", raw, "--length_scale", str(ls), "--sentence_silence", "0.35"], input=l["ru"].encode(), check=True, capture_output=True)
        # pitch shift keeping tempo, trim silence, warm EQ, gentle compression
        sr = 22050
        af = (f"asetrate={int(sr*pitch)},aresample=48000,atempo={1/pitch:.4f},"
              "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
              "highpass=f=70,lowshelf=g=3:f=180,acompressor=threshold=-20dB:ratio=3:attack=5:release=80,loudnorm=I=-16:TP=-1.5")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", af, "-ar", "48000", "-ac", "1", fin], check=True)
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", fin]))
        out.append({"scene": sc["id"], "i": i, "sp": l["sp"], "text": l["ru"], "file": fin, "dur": dur, "hold": l.get("hold", 0)})
        print(sc["id"], i, l["sp"], round(dur, 2), l["ru"])
json.dump(out, open(f"{S}/build/vo.json", "w"), ensure_ascii=False, indent=1)
