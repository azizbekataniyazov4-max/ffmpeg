"""Voice-over with Vosk TTS (ru 0.9 multi). Writes BUILD_DIR/vo.json + wavs.

Usage: python3 tts_vosk.py <script.json> <vosk_model_dir>
"""
import json, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
from vosk_tts import Model, Synth
import timeline as TL

script, model_dir = sys.argv[1], sys.argv[2]
d = json.load(open(script))
os.makedirs(os.path.join(TL.BUILD, "vo"), exist_ok=True)
synth = Synth(Model(model_path=model_dir))
# speaker id, speech rate, pitch factor, extra filter
VOICE = {
    "narrator": (4, 0.88, 0.97, "lowshelf=g=4:f=160,"),
    "future":   (3, 0.92, 0.93, "lowshelf=g=2:f=180,"),
    "hero":     (3, 1.0, 1.13, "highpass=f=120,"),
}
out = []
for sc in d["scenes"]:
    for i, l in enumerate(sc["lines"]):
        spk, rate, pitch, eq = VOICE[l["sp"]]
        raw = os.path.join(TL.BUILD, "vo", f"{sc['id']}_{i}_raw.wav")
        fin = os.path.join(TL.BUILD, "vo", f"{sc['id']}_{i}.wav")
        synth.synth(l["ru"], raw, speaker_id=spk, speech_rate=rate, noise_level=0.6, duration_noise_level=0.6)
        af = (f"asetrate={int(22050 * pitch)},aresample=48000,atempo={1 / pitch:.4f},"
              "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
              f"highpass=f=60,{eq}acompressor=threshold=-20dB:ratio=3:attack=5:release=80,loudnorm=I=-16:TP=-1.5")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", af, "-ar", "48000", "-ac", "1", fin], check=True)
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", fin]))
        out.append({"scene": sc["id"], "i": i, "sp": l["sp"], "text": l["ru"], "file": fin, "dur": dur, "hold": l.get("hold", 0)})
        print(sc["id"], i, l["sp"], round(dur, 2), l["ru"])
json.dump(out, open(os.path.join(TL.BUILD, "vo.json"), "w"), ensure_ascii=False, indent=1)
