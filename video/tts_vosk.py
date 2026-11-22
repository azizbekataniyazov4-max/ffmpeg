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
    "narrator": (4, 1.12, 1.0, "equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=1:g=2.5,"),
    "future":   (3, 1.1, 1.0, "equalizer=f=250:t=q:w=1:g=-1.5,equalizer=f=3000:t=q:w=1:g=2,"),
    "hero":     (3, 1.15, 1.12, "highpass=f=110,equalizer=f=3500:t=q:w=1:g=2,"),
}
out = []
for sc in d["scenes"]:
    for i, l in enumerate(sc["lines"]):
        spk, rate, pitch, eq = VOICE[l["sp"]]
        raw = os.path.join(TL.BUILD, "vo", f"{sc['id']}_{i}_raw.wav")
        fin = os.path.join(TL.BUILD, "vo", f"{sc['id']}_{i}.wav")
        synth.synth(l["ru"], raw, speaker_id=spk, speech_rate=rate, noise_level=0.6, duration_noise_level=0.6)
        shift = f"rubberband=pitch={pitch}:formant=preserved:pitchq=quality," if pitch != 1.0 else ""
        af = (f"aresample=48000,{shift}"
              "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
              f"highpass=f=60,{eq}acompressor=threshold=-18dB:ratio=2:attack=10:release=120,loudnorm=I=-16:TP=-1.5")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", af, "-ar", "48000", "-ac", "1", fin], check=True)
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", fin]))
        out.append({"scene": sc["id"], "i": i, "sp": l["sp"], "text": l["ru"], "file": fin, "dur": dur, "hold": l.get("hold", 0)})
        print(sc["id"], i, l["sp"], round(dur, 2), l["ru"])
json.dump(out, open(os.path.join(TL.BUILD, "vo.json"), "w"), ensure_ascii=False, indent=1)
