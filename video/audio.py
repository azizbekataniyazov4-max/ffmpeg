"""Synthesize SFX + piano score and mix them with the voice lines.

Usage: python3 audio.py <out.wav>
"""
import subprocess
import sys

import numpy as np

import timeline as TL

SR = 48000
tl = TL.build()
DUR = tl["duration"] + 0.2
N = int(DUR * SR)
rng = np.random.default_rng(1)
SC = {s["id"]: s for s in tl["scenes"]}


def T(sid, lt):
    return SC[sid]["start"] + lt


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def lp(x, a):
    """one-pole lowpass, a in (0,1): smaller = darker"""
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def lp_fast(x, cutoff):
    # FFT brickwall-ish lowpass with soft rolloff
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def hp_fast(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (cutoff / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, len(x))


class Bus:
    def __init__(self):
        self.x = np.zeros(N)

    def add(self, t, sig, g=1.0):
        i = int(t * SR)
        if i >= N:
            return
        j = min(N, i + len(sig))
        self.x[i:j] += sig[: j - i] * g


def reverb(x, secs=1.6, mixk=0.25, seed=5):
    r = np.random.default_rng(seed)
    n = int(secs * SR)
    ir = r.normal(0, 1, n) * np.exp(-np.arange(n) / (secs / 5 * SR))
    ir = lp_fast(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
    return x + y * mixk


# ------------------------------------------------------------------ sounds
def knock():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (95 - 30 * t) * t) * env_exp(n, 0.05)
    click = lp_fast(rng.normal(0, 1, n), 1800) * env_exp(n, 0.012)
    return (body * 0.9 + click * 0.8)


def tick():
    n = int(0.03 * SR)
    return hp_fast(rng.normal(0, 1, n), 2500) * env_exp(n, 0.003) * 0.6


def creak(dur=1.1, f0=180):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 + 120 * np.sin(2 * np.pi * 0.7 * t) + 40 * rng.normal(0, 1, n).cumsum() / np.sqrt(n)
    ph = 2 * np.pi * np.cumsum(f) / SR
    # stick-slip pulses
    s = (np.sin(ph) > 0.95).astype(float)
    s = lp_fast(s - s.mean(), 3000)
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
    return s * e * 3


def boom():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * (60 * np.exp(-t * 1.5) + 32) * t) * env_exp(n, 0.6)
    s += lp_fast(rng.normal(0, 1, n), 400) * env_exp(n, 0.15) * 0.6
    return s


def clink(f=2100):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * k * t) * env_exp(n, d) for k, a, d in ((1, 1, 0.35), (1.58, 0.6, 0.22), (2.43, 0.45, 0.15), (3.9, 0.3, 0.08)))
    s[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return s * 0.5


def crackle(dur):
    n = int(dur * SR)
    s = rng.normal(0, 1, n) * (rng.random(n) < 0.02)
    s = hp_fast(s, 1500) + 0.3 * np.sin(2 * np.pi * 100 * np.arange(n) / SR) * (rng.random(n) < 0.5)
    return s * np.hanning(n) * 0.8


def ping():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    a = np.sin(2 * np.pi * 1318.5 * t) * env_exp(n, 0.12)
    b = np.sin(2 * np.pi * 1760 * t) * env_exp(n, 0.15)
    b[: int(0.09 * SR)] = 0
    return (a + b) * 0.35


def chirp():
    n = int(rng.uniform(0.05, 0.12) * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(3000, 4500)
    f = f0 + rng.uniform(-1500, 1500) * t / t[-1]
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(n) * 0.25


def flaps(dur=1.0):
    n = int(dur * SR)
    s = lp_fast(rng.normal(0, 1, n), 1200)
    m = (np.sin(2 * np.pi * 14 * np.arange(n) / SR) > 0.3).astype(float)
    return s * lp_fast(m, 60) * np.hanning(n) * 0.5


def step():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 70 * t) * env_exp(n, 0.03) + lp_fast(rng.normal(0, 1, n), 2500) * env_exp(n, 0.01) * 0.5)


def shutter():
    s = np.zeros(int(0.12 * SR))
    for o in (0, int(0.06 * SR)):
        n = int(0.025 * SR)
        s[o:o + n] += hp_fast(rng.normal(0, 1, n), 1500) * env_exp(n, 0.004)
    return s * 0.9


def clank():
    n = int(1.5 * SR)
    t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * t) * env_exp(n, d) for f, d in ((420, 0.4), (713, 0.25), (1190, 0.12)))
    return lp_fast(s, 1500) * 0.25


def heartbeat():
    s = np.zeros(int(0.6 * SR))
    for o, g in ((0, 1.0), (int(0.22 * SR), 0.7)):
        n = int(0.18 * SR)
        t = np.arange(n) / SR
        s[o:o + n] += np.sin(2 * np.pi * 50 * t) * env_exp(n, 0.05) * g
    return s


def piano(freq, dur=3.0, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for k, a in enumerate((1, 0.5, 0.28, 0.16, 0.09, 0.05), start=1):
        inh = freq * k * (1 + 0.0004 * k * k)
        s += a * np.sin(2 * np.pi * inh * t) * env_exp(n, 1.4 / (1 + 0.5 * k))
    att = int(0.004 * SR)
    s[:att] *= np.linspace(0, 1, att)
    s[-int(0.2 * SR):] *= np.linspace(1, 0, int(0.2 * SR))
    return s * vel * 0.22


def pad(freqs, dur, attack=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for f in freqs:
        for d in (-0.6, 0.0, 0.7):
            s += np.sin(2 * np.pi * (f + d) * t + rng.uniform(0, 6))
    s = lp_fast(s, 1500)
    e = np.minimum(1, t / attack) * np.minimum(1, (dur - t) / 0.8)
    return s * e / len(freqs) * 0.12


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    pc, octv = name[:-1], int(name[-1])
    return 440 * 2 ** ((names[pc] + 12 * (octv - 4)) / 12)


# ------------------------------------------------------------------ build buses
sfx, mus, amb = Bus(), Bus(), Bus()

# S01: ticking, knocks, creak, drone + heartbeat
for k in np.arange(0.1, 4.4, 1.0):
    sfx.add(T("S01", k), tick(), 0.35)
for k in (3.4, 3.75, 4.1):
    sfx.add(T("S01", k), knock(), 0.9)
sfx.add(T("S01", 0.3), ping() * 0.0)
sfx.add(T("S01", 4.65), creak(1.2), 0.25)
n = int(6.5 * SR)
t = np.arange(n) / SR
drone = (np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 82.4 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)) * np.minimum(1, t / 2)
amb.add(T("S01", 3.8), drone * np.minimum(1, (6.5 - t) / 1.0), 0.12)
for k in np.arange(5.9, 10.0, 0.95):
    sfx.add(T("S01", k), heartbeat(), 0.45)
# night room tone
room = lp_fast(rng.normal(0, 1, int(SC["S01"]["end"] * SR)), 300) * 0.05
amb.add(0, room)

# S02: title boom with reverb tail
sfx.add(T("S02", 0.0), reverb(np.concatenate([boom(), np.zeros(SR)]), 2.5, 0.6), 0.8)

# S03: fridge hum, clink, piano
s3 = SC["S03"]
n = int((s3["end"] - s3["start"]) * SR)
t = np.arange(n) / SR
fridge = (np.sin(2 * np.pi * 50 * t) * 0.6 + np.sin(2 * np.pi * 100 * t) * 0.3 + lp_fast(rng.normal(0, 1, n), 200) * 0.6)
amb.add(s3["start"], fridge, 0.045)
sfx.add(T("S03", 1.7), clink(), 0.55)
n = int(1.4 * SR)
slide = lp_fast(rng.normal(0, 1, n), 900) * np.hanning(n) * 0.15
sfx.add(T("S03", 2.0), slide)

# piano: Am - F - C - G arpeggios from S03 start until S04 "version B" cut
prog = [["A2", "E3", "A3", "C4", "E4", "C4"], ["F2", "C3", "F3", "A3", "C4", "A3"], ["C3", "G3", "C4", "E4", "G4", "E4"], ["G2", "D3", "G3", "B3", "D4", "B3"]]
beat = 0.5


def arps(start, end, vel=0.8):
    tt = start
    i = 0
    while tt < end - 0.1:
        ch = prog[(i // 6) % 4]
        nm = ch[i % 6]
        mus.add(tt, piano(note(nm), 2.6, vel * (1.0 if i % 6 == 0 else 0.7)))
        tt += beat
        i += 1


arps(s3["start"] + 0.3, SC["S04"]["start"] + 0.25)
# S04: silence on version B, lamp crackle, notification ping, piano back on "Мы оба"
s4 = SC["S04"]
for k in np.arange(0.3, 3.6, 0.33):
    sfx.add(T("S04", k), crackle(0.18), 0.25)
sfx.add(T("S04", 2.0), ping(), 0.6)
sfx.add(T("S04", 0.25), boom()[: int(0.6 * SR)] * np.linspace(1, 0, int(0.6 * SR)), 0.4)
L1 = s4["lines"][1]
arps(L1["start"] - 0.8, s4["end"] + 0.2, 0.7)

# S05: birds, steps, build-up, gym door
s5 = SC["S05"]
for k in range(14):
    sfx.add(T("S05", rng.uniform(0.0, 4.4)), chirp(), 0.6)
sfx.add(T("S05", 2.9), flaps(1.2), 0.7)
for k in np.arange(1.0, 2.3, 0.42):
    sfx.add(T("S05", k), step(), 0.35)
for k in np.arange(2.3, 4.4, 0.24):
    sfx.add(T("S05", k), step(), 0.45)
sfx.add(T("S05", 4.4), creak(0.7, 260), 0.25)
for i in range(7):
    sfx.add(T("S05", 4.85 + i * 0.165), step() * 0.6, 0.5)
    sfx.add(T("S05", 4.85 + i * 0.165), tick() * 2, 0.4)
# rising music: faster arps + pad
tt = s5["start"]
i = 0
while tt < s5["end"]:
    ch = prog[(i // 8) % 4]
    mus.add(tt, piano(note(ch[i % 6]), 2.0, 0.55 + 0.4 * (tt - s5["start"]) / (s5["end"] - s5["start"])))
    tt += 0.25
    i += 1
mus.add(s5["start"], pad([note("A2"), note("E3"), note("C4")], s5["end"] - s5["start"] + 0.5, 3.0), 1.0)

# S06: gym hum, plate clanks, shutter, final chord
s6 = SC["S06"]
n = int((s6["end"] - s6["start"]) * SR)
gymhum = lp_fast(rng.normal(0, 1, n), 250) * 0.8 + np.sin(2 * np.pi * 120 * np.arange(n) / SR) * 0.15
amb.add(s6["start"], gymhum, 0.05)
for k in (0.6, 2.4, 4.1):
    sfx.add(T("S06", k), reverb(clank(), 1.5, 0.5), 0.35)
sfx.add(T("S06", 1.3), shutter(), 0.8)
chord = sum(piano(note(nm), 5.0, 0.9) for nm in ("A2", "E3", "A3", "C4", "E4", "A4"))
L6 = s6["lines"][0]
mus.add(L6["end"] + 0.1, chord, 0.8)
mus.add(s6["start"], pad([note("A2"), note("E3"), note("A3")], L6["end"] - s6["start"] + 0.5, 1.0), 0.8)

# ------------------------------------------------------------------ voices
vo = Bus()
for sc in tl["scenes"]:
    for l in sc["lines"]:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", l["file"], "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        vo.add(l["start"], x)
vox = reverb(vo.x, 0.9, 0.08, seed=11)

# ducking envelope from the voice
env = np.abs(vo.x)
k = int(0.25 * SR)
env = np.convolve(env, np.ones(k) / k, mode="same")
duck = 1 - 0.55 * np.clip(env / 0.03, 0, 1)

music = reverb(mus.x, 2.2, 0.35, seed=3)
mix = vox * 1.0 + music * duck * 0.55 + reverb(sfx.x, 1.2, 0.18, seed=9) * 0.7 + amb.x * 0.8
mix[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.89
stereo = np.stack([mix, mix], axis=1).astype(np.float32)
out = sys.argv[1]
p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-af", "loudnorm=I=-14:TP=-1.0:LRA=11", "-ar", str(SR), out], input=stereo.tobytes(), check=True)
print("audio", round(DUR, 2), "s ->", out)
