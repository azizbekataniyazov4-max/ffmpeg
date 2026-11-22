"""Timeline shared by the picture renderer and the audio mixer.

All times are in seconds from the start of the video. Scene lengths are driven
by the real duration of each synthesized voice line (build/vo.json).
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("BUILD_DIR", os.path.join(ROOT, "build"))
FPS = 30
W, H = 1080, 1920

# per scene: time before first line, gaps after each line (index -> extra gap), time after last line
SCENE_CFG = {
    "S01": {"pre": 0.6, "gaps": {0: 2.75}, "post": 0.7},
    "S02": {"pre": 0.7, "gaps": {}, "post": 0.9},
    "S03": {"pre": 1.0, "gaps": {0: 0.4, 1: 0.4}, "post": 0.9},
    "S04": {"pre": 0.8, "gaps": {0: 0.95}, "post": 0.7},
    "S05": {"pre": 1.8, "gaps": {}, "post": 2.6},
    "S06": {"pre": 1.0, "gaps": {}, "post": 3.2},
}


def build():
    vo = json.load(open(os.path.join(BUILD, "vo.json")))
    scenes = []
    t = 0.0
    for sid in ["S01", "S02", "S03", "S04", "S05", "S06"]:
        cfg = SCENE_CFG[sid]
        lines = [l for l in vo if l["scene"] == sid]
        start = t
        cur = t + cfg["pre"]
        out = []
        for i, l in enumerate(lines):
            out.append({**l, "start": cur, "end": cur + l["dur"]})
            cur += l["dur"]
            if i < len(lines) - 1:
                cur += cfg["gaps"].get(i, 0.35)
        end = cur + cfg["post"]
        scenes.append({"id": sid, "start": start, "end": end, "lines": out})
        t = end
    return {"scenes": scenes, "duration": t}


if __name__ == "__main__":
    tl = build()
    for s in tl["scenes"]:
        print(s["id"], round(s["start"], 2), round(s["end"], 2), [(round(l["start"], 2), round(l["end"], 2)) for l in s["lines"]])
    print("total", round(tl["duration"], 2))
