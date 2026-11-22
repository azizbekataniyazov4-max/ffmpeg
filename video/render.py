"""Procedural stickman-cinema renderer for «Что если ты встретил себя через 10 лет?».

Usage: python3 render.py <first_frame> <last_frame_exclusive> <out.mp4>
       python3 render.py --still <t> <out.png>
"""
import math
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import timeline as TL

W, H, FPS = TL.W, TL.H, TL.FPS
FONT_DIR = os.environ.get("FONT_DIR", os.path.join(TL.ROOT, "fonts"))

# ---------------------------------------------------------------- palette
BG = (14, 17, 34)
PANEL = (22, 26, 49)
LINE = (43, 49, 87)
FG = (238, 235, 247)
MUTED = (163, 161, 196)
YELLOW = (255, 212, 90)
GREEN = (124, 242, 154)
RED = (255, 96, 96)
ORANGE = (242, 166, 90)
BLUE = (141, 184, 255)
PINK = (242, 138, 192)
WARM = (255, 178, 92)
MOON = (150, 180, 255)
GOLD = (242, 193, 78)
GOLD_D = (176, 122, 30)
SHIRT = (244, 243, 248)
SHIRT_B = (122, 124, 136)
PANTS = (10, 11, 18)
SCREEN = (90, 150, 255)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def dim(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def clamp01(t):
    return max(0.0, min(1.0, t))


def seg01(t, a, b):
    return clamp01((t - a) / (b - a)) if b > a else 1.0


_fonts = {}


def font(name, size):
    key = (name, size)
    if key not in _fonts:
        f = {"display": "Rubik_wght_900.ttf", "display8": "Rubik_wght_800.ttf", "body": "Golos_Text_wght_600.ttf"}[name]
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, f), size)
    return _fonts[key]


# ---------------------------------------------------------------- canvas / camera
class Cam:
    def __init__(self, cx=W / 2, cy=H / 2, z=1.0, sx=0.0, sy=0.0):
        self.cx, self.cy, self.z, self.sx, self.sy = cx, cy, z, sx, sy

    def p(self, x, y):
        return ((x - self.cx) * self.z + W / 2 + self.sx, (y - self.cy) * self.z + H / 2 + self.sy)

    def s(self, v):
        return v * self.z


GS = 4  # glow layer downscale


class Canvas:
    def __init__(self, bg=BG):
        self.img = Image.new("RGB", (W, H), bg)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self.glow = Image.new("RGB", (W // GS, H // GS), (0, 0, 0))
        self.gd = ImageDraw.Draw(self.glow, "RGBA")
        self.cam = Cam()

    # world-space primitives ------------------------------------------------
    def poly(self, pts, fill=None, outline=None, width=1, alpha=255):
        P = [self.cam.p(*q) for q in pts]
        f = (*fill, alpha) if fill else None
        self.d.polygon(P, fill=f, outline=outline, width=max(1, int(self.cam.s(width))) if outline else 0)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1, alpha=255, r=0):
        a, b = self.cam.p(x0, y0), self.cam.p(x1, y1)
        f = (*fill, alpha) if fill else None
        o = (*outline, alpha) if outline else None
        box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        if r:
            self.d.rounded_rectangle(box, radius=self.cam.s(r), fill=f, outline=o, width=max(1, int(self.cam.s(width))) if o else 0)
        else:
            self.d.rectangle(box, fill=f, outline=o, width=max(1, int(self.cam.s(width))) if o else 0)

    def circle(self, x, y, r, fill=None, outline=None, width=1, alpha=255):
        cx, cy = self.cam.p(x, y)
        rr = self.cam.s(r)
        f = (*fill, alpha) if fill else None
        o = (*outline, alpha) if outline else None
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=f, outline=o, width=max(1, int(self.cam.s(width))) if o else 0)

    def ellipse(self, x, y, rx, ry, fill=None, outline=None, width=1, alpha=255):
        cx, cy = self.cam.p(x, y)
        a, b = self.cam.s(rx), self.cam.s(ry)
        f = (*fill, alpha) if fill else None
        o = (*outline, alpha) if outline else None
        self.d.ellipse([cx - a, cy - b, cx + a, cy + b], fill=f, outline=o, width=max(1, int(self.cam.s(width))) if o else 0)

    def line(self, pts, col, w, alpha=255, caps=True):
        P = [self.cam.p(*q) for q in pts]
        ww = max(1, int(self.cam.s(w)))
        self.d.line(P, fill=(*col, alpha), width=ww, joint="curve")
        if caps and ww > 3:
            rr = ww / 2 - 0.5
            for (x, y) in (P[0], P[-1]):
                self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*col, alpha))

    def arc(self, x, y, r, a0, a1, col, w, alpha=255):
        cx, cy = self.cam.p(x, y)
        rr = self.cam.s(r)
        self.d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0, a1, fill=(*col, alpha), width=max(1, int(self.cam.s(w))))

    # glow-space primitives (additive light, blurred later) -----------------
    def gp(self, x, y):
        a = self.cam.p(x, y)
        return (a[0] / GS, a[1] / GS)

    def gcircle(self, x, y, r, col, k=1.0):
        cx, cy = self.gp(x, y)
        rr = self.cam.s(r) / GS
        self.gd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=dim(col, k))

    def gpoly(self, pts, col, k=1.0):
        self.gd.polygon([self.gp(*q) for q in pts], fill=dim(col, k))

    def gline(self, pts, col, w, k=1.0):
        self.gd.line([self.gp(*q) for q in pts], fill=dim(col, k), width=max(1, int(self.cam.s(w) / GS)))

    def grect(self, x0, y0, x1, y1, col, k=1.0, r=0):
        a, b = self.gp(x0, y0), self.gp(x1, y1)
        box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        if r:
            self.gd.rounded_rectangle(box, radius=self.cam.s(r) / GS, fill=dim(col, k))
        else:
            self.gd.rectangle(box, fill=dim(col, k))

    # screen-space text ----------------------------------------------------
    def text(self, xy, s, f, fill, anchor="mm", alpha=255, stroke=0, stroke_fill=(0, 0, 0)):
        self.d.text(xy, s, font=f, fill=(*fill, alpha), anchor=anchor, stroke_width=stroke, stroke_fill=(*stroke_fill, alpha))

    def wtext(self, x, y, s, size, fill, name="display", anchor="mm", alpha=255):
        """world-space text (scales with camera)"""
        px, py = self.cam.p(x, y)
        sz = max(6, int(self.cam.s(size)))
        self.d.text((px, py), s, font=font(name, sz), fill=(*fill, alpha), anchor=anchor)


# ---------------------------------------------------------------- post-processing assets
def _vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
    r = np.sqrt(nx ** 2 * 1.0 + ny ** 2 * 0.75)
    v = 1.0 - 0.55 * np.clip(r - 0.35, 0, 1) ** 1.6
    return v[..., None]


VIG = _vignette()
_rng = np.random.default_rng(7)
GRAIN = [(_rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)) for _ in range(6)]


def finish(cv, fi, glow_k=1.0, grain=5.0, fade=1.0):
    a = np.asarray(cv.img, dtype=np.float32)
    g = cv.glow.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    g2 = cv.glow.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR)
    gl = np.asarray(g, dtype=np.float32) * 0.9 + np.asarray(g2, dtype=np.float32) * 0.55
    a = a + gl * glow_k
    a = a * VIG
    gr = GRAIN[fi % len(GRAIN)]
    gr = np.repeat(np.repeat(gr, 2, axis=0), 2, axis=1)[..., None]
    a = a + gr * grain
    a = a * fade
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- the stickman
def vec(a, facing):
    r = math.radians(a)
    return (math.sin(r) * facing, math.cos(r))


STAND = dict(lean=0, head=0, lt=-4, lk=2, rt=5, rk=0, la=8, le=8, ra=-10, re=10)
SIT = dict(lean=-4, head=0, lt=86, lk=-82, rt=92, rk=-92, la=25, le=55, ra=15, re=60)


def figure(cv, hip, h, facing=1, pose=None, kind="hero", medal=False, phone=False, headphones=False,
           sil=0.0, face=None, rim=None, number=True, phone_glow=0.0, bag=False):
    """Draw a stickman with hip at `hip` (world coords) and total height h."""
    P = dict(STAND)
    if pose:
        P.update(pose)
    hx, hy = hip
    r = h * 0.085
    Lt = h * 0.31
    thigh, shin = h * 0.24, h * 0.235
    uarm, farm = h * 0.165, h * 0.16
    lean = P["lean"]
    nv = vec(180 - lean, facing)
    neck = (hx + nv[0] * Lt, hy + nv[1] * Lt)
    hv = vec(180 - lean - P["head"], facing)
    head = (neck[0] + hv[0] * r * 1.3, neck[1] + hv[1] * r * 1.3)
    sh = (neck[0] + (hx - neck[0]) * 0.1, neck[1] + (hy - neck[1]) * 0.1)

    def limb(base, a1, a2, l1, l2):
        v1 = vec(a1, facing)
        j = (base[0] + v1[0] * l1, base[1] + v1[1] * l1)
        v2 = vec(a1 + a2, facing)
        e = (j[0] + v2[0] * l2, j[1] + v2[1] * l2)
        return j, e

    lk, lf = limb(hip, P["lt"], P["lk"], thigh, shin)
    rk, rf = limb(hip, P["rt"], P["rk"], thigh, shin)
    le, lh = limb(sh, P["la"], P["le"], uarm, farm)
    re, rh = limb(sh, P["ra"], P["re"], uarm, farm)

    skin = mix(FG, (8, 9, 16), sil)
    shirt = SHIRT_B if kind == "b" else SHIRT
    shirt = mix(shirt, (16, 18, 30), sil)
    pants = PANTS
    rimc = mix(FG, (40, 44, 70), sil)
    if rim is not None:
        rimc = rim
    lw = h * 0.05
    aw = h * 0.036

    def leg(k, f):
        cv.line([hip, k, f], rimc, lw + h * 0.012)
        cv.line([hip, k, f], pants, lw)
        # shoe
        cv.line([f, (f[0] + facing * h * 0.045, f[1])], rimc, h * 0.03)

    def arm(e, hnd):
        cv.line([sh, e, hnd], skin, aw)
        cv.circle(hnd[0], hnd[1], aw * 0.62, fill=skin)

    # back limbs
    leg(lk, lf)
    arm(le, lh)
    if bag:
        bx, by = lh
        cv.rect(bx - h * 0.1, by + h * 0.01, bx + h * 0.1, by + h * 0.13, fill=(30, 34, 60), outline=rimc, width=h * 0.006, r=h * 0.03)
        cv.line([(bx - h * 0.04, by + h * 0.02), (bx, by), (bx + h * 0.04, by + h * 0.02)], rimc, h * 0.008)
    # torso tank top
    tx, ty = neck[0] - hx, neck[1] - hy
    L = math.hypot(tx, ty) or 1
    px, py = -ty / L, tx / L
    ws, wh = h * 0.085, h * 0.075
    top = (neck[0] + (hx - neck[0]) * 0.04, neck[1] + (hy - neck[1]) * 0.04)
    bot = (hx + (neck[0] - hx) * -0.02, hy + (neck[1] - hy) * -0.02)
    torso = [(top[0] + px * ws, top[1] + py * ws), (top[0] - px * ws, top[1] - py * ws),
             (bot[0] - px * wh, bot[1] - py * wh), (bot[0] + px * wh, bot[1] + py * wh)]
    # rim outline
    cv.poly(torso, fill=rimc)
    inner = [(top[0] + px * (ws - h * 0.008), top[1] + py * (ws - h * 0.008) + h * 0.004),
             (top[0] - px * (ws - h * 0.008), top[1] - py * (ws - h * 0.008) + h * 0.004),
             (bot[0] - px * (wh - h * 0.008), bot[1] - py * (wh - h * 0.008) - h * 0.004),
             (bot[0] + px * (wh - h * 0.008), bot[1] + py * (wh - h * 0.008) - h * 0.004)]
    cv.poly(inner, fill=shirt)
    # neckline scoop
    cv.ellipse(top[0], top[1] + h * 0.006, h * 0.03, h * 0.022, fill=skin)
    if kind == "b" and sil < 0.5:
        cv.ellipse(hx + (neck[0] - hx) * 0.35 + h * 0.02, hy + (neck[1] - hy) * 0.35, h * 0.018, h * 0.012, fill=(100, 98, 92), alpha=160)
    if number:
        mx = hx + (neck[0] - hx) * 0.47
        my = hy + (neck[1] - hy) * 0.47
        numc = mix((14, 14, 20), (40, 40, 52), sil) if kind != "b" else (52, 52, 60)
        cv.wtext(mx, my, "21", h * 0.085, numc, name="display")
    # hips
    cv.line([(hx - px * wh * 0.8, hy - py * wh * 0.8), (hx + px * wh * 0.8, hy + py * wh * 0.8)], pants, h * 0.045)
    if headphones:
        cv.arc(neck[0], neck[1] + h * 0.005, h * 0.045, 10, 170, mix((250, 250, 255), (30, 30, 40), sil), h * 0.014)
    if medal:
        mc = (hx + (neck[0] - hx) * 0.62, hy + (neck[1] - hy) * 0.62)
        cv.line([(neck[0] - px * h * 0.03, neck[1] - py * h * 0.03), mc], mix((210, 50, 60), (30, 10, 14), sil * 0.8), h * 0.012)
        cv.line([(neck[0] + px * h * 0.03, neck[1] + py * h * 0.03), mc], mix((60, 100, 230), (10, 14, 40), sil * 0.8), h * 0.012)
        draw_medal(cv, mc[0], mc[1], h * 0.034, glint=1.0)
    # front limbs
    leg(rk, rf)
    if phone:
        cv.rect(rh[0] - h * 0.018, rh[1] - h * 0.05, rh[0] + h * 0.018, rh[1] + h * 0.01, fill=(20, 22, 30), outline=(60, 64, 80), width=h * 0.004, r=h * 0.006)
        if phone_glow > 0:
            cv.rect(rh[0] - h * 0.014, rh[1] - h * 0.045, rh[0] + h * 0.014, rh[1] + h * 0.004, fill=SCREEN, alpha=int(255 * phone_glow))
            cv.gcircle(rh[0], rh[1] - h * 0.02, h * 0.07, SCREEN, 0.7 * phone_glow)
            cv.gcircle(head[0], head[1], r * 1.3, SCREEN, 0.45 * phone_glow)
    arm(re, rh)
    # head
    cv.circle(head[0], head[1], r + h * 0.006, fill=rimc)
    cv.circle(head[0], head[1], r, fill=skin)
    if phone_glow > 0:
        cv.circle(head[0], head[1], r, fill=SCREEN, alpha=int(70 * phone_glow))
    if face is not None and sil < 0.6:
        gx, gy = face.get("gaze", (facing, 0))
        ex = head[0] + facing * r * 0.25 + gx * r * 0.12
        ey = head[1] - r * 0.05 + gy * r * 0.12
        for o in (-1, 1):
            cv.circle(ex + o * r * 0.28, ey, r * 0.1, fill=(20, 20, 30))
    return dict(head=head, neck=neck, hip=hip, lh=lh, rh=rh, r=r, sh=sh)


def draw_medal(cv, x, y, r, glint=0.0, alpha=255, squash=1.0, face=True):
    rx = r * max(0.08, abs(squash))
    cv.ellipse(x, y, rx * 1.12, r * 1.12, fill=GOLD_D, alpha=alpha)
    cv.ellipse(x, y, rx, r, fill=GOLD, alpha=alpha)
    if face and abs(squash) > 0.3:
        cv.ellipse(x, y, rx * 0.68, r * 0.68, outline=GOLD_D, width=r * 0.09, alpha=alpha)
        cv.wtext(x, y + r * 0.02, "1", r * 0.95, GOLD_D, name="display", alpha=alpha)
    cv.ellipse(x - rx * 0.35, y - r * 0.4, rx * 0.22, r * 0.14, fill=(255, 240, 190), alpha=int(alpha * 0.8))
    if glint > 0:
        cv.gcircle(x, y, r * 1.6, GOLD, 0.6 * glint)
        s = r * 2.2 * glint
        cv.gline([(x - rx * 0.4 - s, y - r * 0.4), (x - rx * 0.4 + s, y - r * 0.4)], (255, 245, 210), r * 0.18, 1.0)
        cv.gline([(x - rx * 0.4, y - r * 0.4 - s), (x - rx * 0.4, y - r * 0.4 + s)], (255, 245, 210), r * 0.18, 1.0)


# ---------------------------------------------------------------- overlays
def plate(cv, text, col, t_in, alpha=1.0, y=170):
    """Glowing chip in the style of the script's tag chips."""
    a = ease(t_in / 0.35) * alpha
    if a <= 0:
        return
    f = font("display8", 46)
    tw = cv.d.textlength(text, font=f)
    x0 = 70
    x1 = x0 + tw + 64
    dy = (1 - ease(t_in / 0.35)) * -20
    box = [x0, y + dy, x1, y + 84 + dy]
    cv.d.rounded_rectangle(box, radius=20, fill=(12, 14, 26, int(200 * a)), outline=(*col, int(255 * a)), width=5)
    cv.gd.rounded_rectangle([b / GS for b in box], radius=5, outline=dim(col, 0.8 * a), width=3)
    cv.text(((x0 + x1) / 2, y + 44 + dy), text, f, col, alpha=int(255 * a))


def wrap(text, f, maxw, d):
    words = text.split()
    lines, cur = [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if d.textlength(test, font=f) <= maxw:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


SPK_COL = {"narrator": YELLOW, "future": ORANGE, "hero": GREEN}


def subtitle(cv, t, lines, y=1580):
    for l in lines:
        if l["start"] - 0.08 <= t <= l["end"] + 0.25:
            a = min(seg01(t, l["start"] - 0.08, l["start"] + 0.12), 1 - seg01(t, l["end"] + 0.05, l["end"] + 0.25))
            f = font("body", 60)
            rows = wrap(l["text"], f, 900, cv.d)
            lh = 80
            y0 = y - (len(rows) - 1) * lh / 2
            col = SPK_COL[l["sp"]]
            # speaker accent bar
            for i, row in enumerate(rows):
                cv.text((W / 2, y0 + i * lh), row, f, FG, alpha=int(255 * a), stroke=7, stroke_fill=(6, 7, 14))
            cv.d.rounded_rectangle([W / 2 - 40, y0 - 78, W / 2 + 40, y0 - 68], radius=5, fill=(*col, int(230 * a)))


# ---------------------------------------------------------------- shared set pieces
def hexagon(cx, cy, r, rot=0):
    return [(cx + r * math.cos(math.radians(60 * i + rot)), cy + r * math.sin(math.radians(60 * i + rot))) for i in range(6)]


def kitchen(cv, t, lamp=1.0, swing=0.0, dawn=0.0, cup=True, medal_x=None, medal_a=255, medal_glint=0.4, steam=True):
    """Side view of the night kitchen. Table centre at x=540, top y=1180."""
    wall = mix((20, 22, 44), (60, 52, 70), dawn * 0.6)
    cv.rect(-600, -400, 1700, 1330, fill=wall)
    # tiles line
    for i in range(-6, 18):
        cv.line([(i * 110, 1000), (i * 110, 1330)], dim(wall, 0.82), 3, caps=False)
    cv.line([(-600, 1000), (1700, 1000)], dim(wall, 0.8), 4, caps=False)
    # window
    sky = mix((12, 16, 40), (250, 170, 120), dawn)
    sky2 = mix((24, 30, 70), (255, 215, 150), dawn)
    cv.rect(700, 430, 1000, 820, fill=(10, 10, 20))
    for k in range(8):
        cv.rect(712, 442 + k * 47, 988, 442 + (k + 1) * 47, fill=mix(sky, sky2, k / 7))
    if dawn < 0.6:
        cv.circle(905, 520, 34, fill=(220, 228, 255), alpha=int(255 * (1 - dawn / 0.6)))
        cv.gcircle(905, 520, 70, MOON, 0.6 * (1 - dawn))
    else:
        cv.gcircle(850, 800, 160, (255, 170, 100), (dawn - 0.6) * 1.5)
    cv.line([(850, 442), (850, 808)], (10, 10, 20), 10, caps=False)
    cv.line([(712, 625), (988, 625)], (10, 10, 20), 10, caps=False)
    # floor
    cv.rect(-600, 1330, 1700, 2400, fill=mix((16, 16, 28), (40, 34, 44), dawn * 0.5))
    cv.line([(-600, 1330), (1700, 1330)], dim(LINE, 1.1), 4, caps=False)
    # lamp: cord from ceiling, pendulum swing
    pivot = (540, -200)
    ang = math.radians(swing)
    Lc = 960
    lx, ly = pivot[0] + math.sin(ang) * Lc, pivot[1] + math.cos(ang) * Lc
    cv.line([pivot, (lx, ly)], (8, 8, 14), 5, caps=False)
    # light cone
    if lamp > 0.02:
        cone = [(lx - 50, ly + 40), (lx + 50, ly + 40), (lx + 420 + math.sin(ang) * 300, 1330), (lx - 420 + math.sin(ang) * 300, 1330)]
        cv.poly(cone, fill=WARM, alpha=int(34 * lamp))
        cv.gpoly(cone, WARM, 0.14 * lamp)
        cv.ellipse(lx + math.sin(ang) * 300, 1335, 470, 40, fill=WARM, alpha=int(30 * lamp))
    shade = [(lx - 30, ly - 10), (lx + 30, ly - 10), (lx + 95, ly + 55), (lx - 95, ly + 55)]
    cv.poly(shade, fill=(30, 30, 46))
    cv.line([(lx - 95, ly + 55), (lx + 95, ly + 55)], (70, 70, 96), 5, caps=False)
    cv.ellipse(lx, ly + 58, 34, 16, fill=mix((60, 50, 40), (255, 236, 190), lamp))
    if lamp > 0.02:
        cv.gcircle(lx, ly + 60, 90, WARM, 1.0 * lamp)
        cv.gcircle(lx, ly + 60, 30, (255, 240, 210), 1.2 * lamp)
    # table
    lit = 0.55 + 0.45 * lamp
    cv.rect(300, 1180, 780, 1205, fill=dim((96, 62, 44), lit))
    cv.rect(300, 1176, 780, 1184, fill=dim((140, 96, 66), lit))
    cv.rect(330, 1205, 350, 1330, fill=dim((70, 46, 32), lit))
    cv.rect(730, 1205, 750, 1330, fill=dim((70, 46, 32), lit))
    if cup:
        cv.rect(610, 1128, 660, 1176, fill=dim((220, 220, 232), lit), r=8)
        cv.arc(668, 1150, 16, -90, 90, dim((220, 220, 232), lit), 7)
        if steam:
            for k in range(3):
                pts = []
                for j in range(14):
                    yy = 1115 - j * 9
                    ph = t * 2.2 + k * 2.1
                    xx = 622 + k * 13 + math.sin(j * 0.45 + ph) * (4 + j * 0.8)
                    pts.append((xx, yy - ((t * 30 + k * 20) % 20)))
                cv.line(pts, (230, 230, 240), 4, alpha=int(60 * lit))
    if medal_x is not None and medal_a > 0:
        draw_medal(cv, medal_x, 1168, 20, glint=medal_glint, alpha=medal_a, squash=0.35)
    return (lx, ly)


def stools(cv, lamp=1.0):
    lit = 0.55 + 0.45 * lamp
    for sx in (190, 890):
        cv.rect(sx - 70, 1215, sx + 70, 1235, fill=dim((84, 70, 100), lit), r=6)
        cv.line([(sx - 55, 1235), (sx - 70, 1330)], dim((60, 50, 76), lit), 10)
        cv.line([(sx + 55, 1235), (sx + 70, 1330)], dim((60, 50, 76), lit), 10)


# ================================================================ SCENES
def scene_S01(cv, t, sc, fi):
    L0, L1 = sc["lines"]
    knocks = [3.4, 3.75, 4.1]
    if t < 2.0:  # -------- phone on nightstand
        cv.cam = Cam(540, 1250, 1.0 + 0.05 * t)
        cv.rect(-200, -200, 1300, 2200, fill=(10, 12, 26))
        # moonlight from window
        cv.poly([(-200, 200), (300, 200), (1100, 1500), (500, 1500)], fill=MOON, alpha=14)
        cv.gpoly([(-200, 200), (300, 200), (1100, 1500), (500, 1500)], MOON, 0.05)
        # nightstand top (perspective)
        cv.poly([(80, 1180), (1000, 1180), (1180, 1700), (-100, 1700)], fill=(52, 36, 32))
        cv.poly([(-100, 1700), (1180, 1700), (1180, 1760), (-100, 1760)], fill=(36, 24, 22))
        # phone
        on = 0.0 if t < 0.3 else (1.0 if t > 0.4 else (t - 0.3) / 0.1)
        flick = 1.0 + 0.25 * math.exp(-(t - 0.35) * 8) if t > 0.3 else 1
        ph = [(300, 1240), (780, 1240), (860, 1600), (220, 1600)]
        cv.poly([(q[0], q[1] + 14) for q in ph], fill=(0, 0, 0), alpha=140)
        cv.poly(ph, fill=(18, 20, 28))
        scr = [(318, 1256), (762, 1256), (836, 1586), (244, 1586)]
        cv.poly(scr, fill=mix((14, 16, 22), (40, 70, 140), on))
        if on > 0:
            cv.gpoly(scr, SCREEN, 0.55 * on * flick)
            cv.poly([(240, 1700), (840, 1700), (1000, 1180), (80, 1180)], fill=SCREEN, alpha=int(20 * on))
            px, py = cv.cam.p(540, 1400)
            sz = int(cv.cam.s(150))
            cv.d.text((px, py), "03:00", font=font("display8", sz), fill=(235, 242, 255, int(255 * on)), anchor="mm")
            cv.d.text((px, py + cv.cam.s(110)), "СУББОТА", font=font("body", int(cv.cam.s(40))), fill=(200, 215, 255, int(200 * on)), anchor="mm")
    elif t < 4.5:  # -------- corridor: door, keys jumping
        shake = 0.0
        for k in knocks:
            if k <= t < k + 0.18:
                shake = (1 - (t - k) / 0.18)
        cv.cam = Cam(540, 960, 1.04 + 0.03 * (t - 2), sx=random.uniform(-1, 1) * 14 * shake, sy=random.uniform(-1, 1) * 10 * shake)
        cv.rect(-200, -200, 1300, 2200, fill=(16, 18, 36))
        cv.rect(-200, 1500, 1300, 2200, fill=(12, 12, 24))
        cv.line([(-200, 1500), (1300, 1500)], (36, 40, 70), 5)
        # moon shaft across the floor
        shaft = [(-200, 300), (60, 260), (900, 1900), (300, 1900)]
        cv.poly(shaft, fill=MOON, alpha=22)
        cv.gpoly(shaft, MOON, 0.08)
        # door
        dx = 300
        cv.rect(dx - 20, 520, dx + 440, 1505, fill=(30, 26, 40))
        cv.rect(dx, 540, dx + 420, 1500, fill=(54, 44, 58))
        for (a, b) in ((600, 960), (1020, 1440)):
            cv.rect(dx + 50, a, dx + 370, b, outline=(40, 32, 46), width=8)
        cv.circle(dx + 370, 1050, 16, fill=(200, 180, 120))
        # light under the door, pulses on each knock
        k_on = 0.0
        for k in knocks:
            if t >= k:
                k_on = 1.0
        cv.rect(dx, 1492, dx + 420, 1500, fill=WARM, alpha=int(200 * (0.4 + 0.6 * k_on)))
        cv.grect(dx, 1490, dx + 420, 1504, WARM, 0.8)
        # hook with keys
        kx, ky = 860, 860
        cv.rect(kx - 40, ky - 20, kx + 40, ky, fill=(80, 76, 90), r=6)
        jump = 0.0
        for k in knocks:
            if k <= t < k + 0.3:
                jump = math.sin((t - k) / 0.3 * math.pi) * 30
        cv.line([(kx, ky), (kx, ky + 40 - jump)], (160, 160, 175), 5)
        cv.circle(kx, ky + 55 - jump, 18, outline=(190, 190, 205), width=5)
        for j, a in enumerate((-30, 10, 40)):
            ang = math.radians(a + jump * 0.8 * (1 if j % 2 else -1))
            x0, y0 = kx, ky + 70 - jump
            x1, y1 = x0 + math.sin(ang) * 60, y0 + math.cos(ang) * 60
            cv.line([(x0, y0), (x1, y1)], (200, 186, 130) if j != 1 else (180, 180, 195), 9)
            cv.circle(x1, y1, 7, fill=(200, 186, 130) if j != 1 else (180, 180, 195))
        if jump > 0:
            cv.gcircle(kx, ky + 80, 50, MOON, 0.25 * jump / 30)
        # knock shockwaves
        for k in knocks:
            if k <= t < k + 0.35:
                q = (t - k) / 0.35
                cv.circle(dx + 210, 900, 60 + q * 260, outline=(255, 220, 160), width=6, alpha=int(150 * (1 - q)))
    elif t < 5.9:  # -------- hero opens the door, warm strip widens
        q = ease(seg01(t, 4.7, 5.7))
        cv.cam = Cam(540, 1000, 1.0 + 0.04 * (t - 4.5))
        cv.rect(-200, -200, 1300, 2200, fill=(14, 16, 32))
        cv.rect(-200, 1520, 1300, 2200, fill=(11, 11, 22))
        # door frame
        fx0, fx1 = 560, 900
        cv.rect(fx0 - 20, 520, fx1 + 20, 1525, fill=(28, 24, 38))
        # warm opening
        open_w = (fx1 - fx0) * q
        cv.rect(fx0, 540, fx0 + open_w, 1520, fill=(255, 196, 120))
        cv.grect(fx0, 540, fx0 + open_w, 1520, WARM, 1.0)
        # door leaf (rotating toward camera)
        leaf = [(fx0 + open_w, 540), (fx1 + (fx1 - fx0) * 0.3 * q, 500), (fx1 + (fx1 - fx0) * 0.3 * q, 1560), (fx0 + open_w, 1520)]
        if q < 0.999:
            cv.poly([(fx0 + open_w, 540), (fx1, 540), (fx1, 1520), (fx0 + open_w, 1520)], fill=(54, 44, 58))
        else:
            cv.poly(leaf, fill=(44, 36, 50))
        # light spill on floor
        if q > 0:
            spill = [(fx0, 1520), (fx0 + open_w, 1520), (fx0 + open_w * 2.2 - 200, 2200), (fx0 - 500 * q, 2200)]
            cv.poly(spill, fill=WARM, alpha=int(70 * q))
            cv.gpoly(spill, WARM, 0.25 * q)
        # hero (from behind-left), rim lit by the opening
        reach = ease(seg01(t, 4.5, 4.8))
        figure(cv, (430, 1210), 700, facing=1,
               pose=dict(ra=-10 - 60 * reach + 20 * q, re=10 + 10 * reach, la=-5, le=10, lean=3),
               kind="hero", headphones=True, rim=mix((60, 64, 100), (255, 200, 140), q), face=None)
    else:  # -------- low angle: tall silhouette in the doorway, medal glints
        lt = t - 5.9
        z = 1.0 + 0.06 * lt
        cv.cam = Cam(540, 900 + lt * 10, z)
        for k in range(14):
            cv.rect(-300, -400 + k * 200, 1400, -200 + k * 200, fill=mix((255, 214, 150), (200, 110, 60), k / 13))
        cv.gcircle(540, 500, 700, WARM, 0.35)
        # door frame edges (dark, perspective from low angle)
        cv.poly([(-300, -400), (120, -400), (170, 2400), (-300, 2400)], fill=(20, 16, 26))
        cv.poly([(960, -400), (1400, -400), (1400, 2400), (910, 2400)], fill=(20, 16, 26))
        cv.poly([(-300, 1700), (1400, 1700), (1400, 2400), (-300, 2400)], fill=(30, 22, 30))
        # silhouette: big, from below
        step = ease(seg01(lt, 0, 0.6))
        info = figure(cv, (540, 1060 - step * 10), 1500, facing=-1,
                      pose=dict(lean=-2, la=18, le=8, ra=-18, re=10, lt=-6, rt=6, head=-4 + 3 * math.sin(lt * 2)),
                      kind="future", medal=True, sil=0.93, rim=(255, 226, 180), number=False)
        # focus ring on the medal
        mx = info["hip"][0] + (info["neck"][0] - info["hip"][0]) * 0.62
        my = info["hip"][1] + (info["neck"][1] - info["hip"][1]) * 0.62
        fr = seg01(lt, 0.5, 1.4)
        if 0 < fr < 1:
            cv.circle(mx, my, 260 - 190 * ease(fr), outline=(255, 255, 255), width=4, alpha=int(200 * math.sin(fr * math.pi)))
        draw_medal(cv, mx, my, 1500 * 0.034, glint=0.6 + 0.6 * max(0, math.sin(lt * 3)))
    subtitle(cv, t + sc["start"], sc["lines"])


def scene_S02(cv, t, sc, fi):
    dur = sc["end"] - sc["start"]
    cv.cam = Cam(540, 960, 1.0 + 0.02 * t)
    cv.rect(-100, -100, 1200, 2100, fill=(10, 10, 20))
    cv.gcircle(540, 900, 560, (90, 50, 28), 0.8)
    sq = math.cos(t * 1.4)
    draw_medal(cv, 540, 880, 300, glint=0.0, alpha=60, squash=sq, face=abs(sq) > 0.3)
    hit = seg01(t, 0, 0.45)
    s = 1.12 - 0.12 * ease(hit)
    a = ease(hit)
    rows = ["ЧТО ЕСЛИ", "ТЫ ВСТРЕТИЛ", "СЕБЯ ЧЕРЕЗ"]
    for i, r in enumerate(rows):
        f = font("display", int(110 * s))
        cv.text((W / 2, 640 + i * 132 * s), r, f, FG, alpha=int(255 * a), stroke=4, stroke_fill=(8, 8, 16))
    f = font("display", int(230 * s))
    cv.text((W / 2, 1110 + 10 * s), "10 ЛЕТ?", f, ORANGE, alpha=int(255 * a), stroke=6, stroke_fill=(30, 12, 0))
    cv.gd.text((W / 2 / GS, 1110 / GS), "10 ЛЕТ?", font=font("display", int(230 * s / GS)), fill=dim(ORANGE, 0.9 * a), anchor="mm")
    if t < 0.12:
        cv.d.rectangle([0, 0, W, H], fill=(255, 230, 200, int(180 * (1 - t / 0.12))))


def scene_S03(cv, t, sc, fi):
    L0, L1, L2 = sc["lines"]
    T0 = lambda l: l["start"] - sc["start"]
    T1 = lambda l: l["end"] - sc["start"]
    a_end = T0(L0) + 0.6 * (T1(L0) - T0(L0))
    b_end = T1(L0) + 0.2
    c_end = T1(L1) + 0.25
    swing = 2.0 * math.sin(t * 1.3)
    # medal motion: placed at 1.6-2.0, slides 2.1-3.2
    place = seg01(t, 1.2, 1.7)
    slide = ease(seg01(t, 2.0, 3.4))
    medal_x = 700 - 300 * slide if t >= 1.7 else None
    if t < a_end:  # -------- wide: two at the table
        cv.cam = Cam(540, 1050, 0.92 + 0.012 * t)
        kitchen(cv, t, lamp=1.0, swing=swing, medal_x=medal_x, medal_glint=0.5)
        stools(cv)
        figure(cv, (190, 1205), 560, facing=1, pose=dict(SIT, ra=45, re=50, la=30, le=60), kind="hero", headphones=True,
               face={"gaze": (1, 0)}, phone=False)
        # future: reaches to place medal, then pushes it
        reach = math.sin(place * math.pi) if t < 1.7 else (0.7 if t < 3.4 else 0.7 * (1 - seg01(t, 3.4, 3.8)))
        figure(cv, (890, 1190), 620, facing=-1,
               pose=dict(SIT, ra=20 + 70 * reach, re=10 + 10 * (1 - reach), la=25, le=55, head=4),
               kind="future", medal=t < 1.45, face={"gaze": (-1, 0.2)})
        plate(cv, "2036 · ЧЕМПИОН", YELLOW, t)
    elif t < b_end:  # -------- insert: sports bag by the door
        lt = t - a_end
        cv.cam = Cam(540, 1100, 1.15 + 0.03 * lt)
        cv.rect(-200, -200, 1300, 2300, fill=(22, 22, 42))
        cv.rect(-200, 1420, 1300, 2300, fill=(18, 16, 28))
        cv.rect(700, 300, 1100, 1430, fill=(46, 38, 54))  # door edge
        cv.poly([(-200, 1420), (1300, 1420), (1300, 2300), (-200, 2300)], fill=WARM, alpha=14)
        # bag
        cv.rect(150, 1040, 830, 1430, fill=(30, 36, 80), outline=(70, 80, 140), width=8, r=90)
        cv.line([(200, 1110), (780, 1110)], (110, 120, 180), 8)
        cv.arc(490, 1040, 180, 180, 360, (60, 66, 120), 22)
        cv.wtext(490, 1290, "21", 120, (230, 230, 240))
        # wrestling shoes sticking out
        for k, (x, a) in enumerate(((330, -20), (520, 15))):
            ang = math.radians(a)
            pts = [(x, 1060), (x + math.sin(ang) * 210, 1060 - math.cos(ang) * 210)]
            cv.line(pts, (210, 40, 50), 70)
            cv.line([(pts[0][0] + 16, pts[0][1] - 30), (pts[1][0] + 10, pts[1][1] + 26)], (245, 245, 250), 10)
            for j in range(4):
                yy = 1060 - 40 - j * 36
                cv.line([(x + math.sin(ang) * (40 + j * 36) - 20, yy), (x + math.sin(ang) * (40 + j * 36) + 20, yy - 6)], (250, 250, 255), 5)
        cv.gcircle(500, 1100, 300, WARM, 0.12)
    elif t < c_end:  # -------- close-up: your face, doubt
        lt = t - b_end
        cv.cam = Cam(540, 960, 1.0 + 0.04 * lt)
        cv.rect(-200, -200, 1300, 2300, fill=(28, 26, 48))
        cv.gcircle(900, 300, 300, WARM, 0.35)
        cv.circle(880, 330, 160, fill=(70, 56, 60), alpha=120)
        # shoulders + tank straps
        cv.ellipse(540, 1700, 520, 300, fill=FG)
        cv.poly([(330, 1480), (750, 1480), (900, 2200), (180, 2200)], fill=(60, 64, 100))
        cv.poly([(345, 1500), (735, 1500), (880, 2200), (200, 2200)], fill=SHIRT)
        cv.ellipse(540, 1490, 140, 90, fill=FG)
        cv.arc(540, 1460, 200, 15, 165, (250, 250, 255), 30)  # headphones round the neck
        cv.circle(350, 1530, 34, fill=(250, 250, 255))
        cv.circle(730, 1530, 34, fill=(250, 250, 255))
        # head
        cv.rect(490, 1260, 590, 1460, fill=FG)
        cv.circle(540, 1000, 360, fill=(60, 64, 100))
        cv.circle(540, 1000, 350, fill=FG)
        cv.circle(540, 1000, 350, fill=WARM, alpha=26)
        gaze = 0.9
        for o in (-1, 1):
            ex = 540 + o * 125 + gaze * 20
            cv.ellipse(ex, 990, 44, 52, fill=(250, 250, 255))
            cv.circle(ex + gaze * 10, 996, 24, fill=(18, 18, 28))
            cv.circle(ex + gaze * 10 - 8, 986, 7, fill=(255, 255, 255))
            # brows: one raised (doubt)
            ra = 30 if o == 1 else 8
            cv.line([(ex - 50, 900 - ra), (ex + 50, 895 - ra - (10 if o == 1 else -4))], (30, 30, 44), 16)
        talk = (L1["start"] - sc["start"] <= t <= L1["end"] - sc["start"])
        mo = abs(math.sin(t * 18)) * 18 if talk else 0
        cv.ellipse(560, 1170, 60, 10 + mo, fill=(30, 26, 40))
    else:  # -------- over the shoulder: future nods slowly
        lt = t - c_end
        cv.cam = Cam(700, 1000, 1.45 + 0.03 * lt)
        kitchen(cv, t, lamp=1.0, swing=swing * 1.5, medal_x=400, medal_glint=0.6)
        stools(cv)
        nod = math.sin(lt * 2.6) * 6 * (1 - seg01(lt, 2.5, 4))
        figure(cv, (890, 1190), 620, facing=-1, pose=dict(SIT, ra=30, re=50, la=25, le=55, head=6 + nod, lean=-6),
               kind="future", medal=False, face={"gaze": (-1, 0.1 + nod / 30)})
        # hero's shoulder silhouette in the foreground
        cv.cam = Cam(700, 1000, 1.45)
        cv.ellipse(300, 1150, 230, 400, fill=(12, 12, 22))
        cv.circle(320, 760, 160, fill=(12, 12, 22))
        cv.arc(320, 760, 162, 260, 360, (70, 60, 70), 6)
    subtitle(cv, t + sc["start"], sc["lines"])


def flicker_state(t):
    """Lamp flicker pattern for S04: returns (lamp_level, versionB_visible)."""
    if t < 0.25:
        return 1.0, False
    if t < 3.4:
        q = t - 0.25
        # irregular flicker cycles of varied length
        pat = [0.22, 0.12, 0.35, 0.1, 0.5, 0.14, 0.28, 0.1, 0.45, 0.12, 0.3, 0.1]
        acc, on = 0, True
        for p in pat:
            if q < acc + p:
                return (1.0 if on else 0.12), (not on) or (q > 1.6 and random.random() < 0.0)
            acc += p
            on = not on
        return 0.15, True
    return None


def scene_S04(cv, t, sc, fi):
    L0, L1 = sc["lines"]
    t0 = L0["start"] - sc["start"]
    a_end = t0 + 3.3
    b_end = t0 + 4.6
    c_end = L0["end"] - sc["start"] + 0.35
    if t < a_end:  # -------- the stool opposite, lamp flickers between the two versions
        st = flicker_state(t)
        lamp, isB = st if st else (0.15, True)
        cv.cam = Cam(760, 1020, 1.35 + 0.02 * t)
        kitchen(cv, t, lamp=lamp, swing=0.6 * math.sin(t * 3), medal_x=400 if not isB else None, medal_glint=0.3 * lamp, steam=False)
        stools(cv, lamp)
        if isB:
            figure(cv, (890, 1200), 600, facing=-1,
                   pose=dict(SIT, lean=-22, head=-20, ra=60, re=70, la=40, le=70), kind="b", phone=True, phone_glow=1.0,
                   face={"gaze": (0, 1)}, rim=(70, 80, 120))
        else:
            figure(cv, (890, 1190), 620, facing=-1, pose=dict(SIT, ra=30, re=50, la=25, le=55, head=4), kind="future",
                   face={"gaze": (-1, 0)}, rim=dim(FG, 0.4 + 0.6 * lamp))
        plate(cv, "2036 · ВЕРСИЯ Б" if (isB or t > 0.6) else "2036 · ЧЕМПИОН", RED if (isB or t > 0.6) else YELLOW, max(t - 0.25, 0) if t > 0.25 else 1.0)
    elif t < b_end:  # -------- insert: medal on the table dims and disappears
        lt = t - a_end
        q = seg01(lt, 0.15, 1.1)
        cv.cam = Cam(420, 1160, 3.4 + 0.1 * lt)
        kitchen(cv, t, lamp=0.5 * (1 - q) + 0.12, swing=0, medal_x=None, steam=False)
        draw_medal(cv, 420, 1166, 22, glint=0.5 * (1 - q), alpha=int(255 * (1 - ease(q))), squash=0.35)
    elif t < c_end:  # -------- your eyes dart between the two versions
        lt = t - b_end
        cv.cam = Cam(540, 960, 1.0)
        cv.rect(-100, -100, 1200, 2100, fill=(14, 16, 30))
        g = math.sin(lt * 7.5)
        g = 1 if g > 0.2 else (-1 if g < -0.2 else g * 5)
        cv.gcircle(540 + g * 600, 960, 300, SCREEN if g < 0 else WARM, 0.25)
        cv.rect(-100, 660, 1200, 1300, fill=FG)
        cv.rect(-100, 660, 1200, 1300, fill=SCREEN if g < 0 else WARM, alpha=26)
        for o in (-1, 1):
            ex = 540 + o * 250
            cv.ellipse(ex, 990, 170, 150, fill=(250, 250, 255))
            cv.circle(ex + g * 80, 1000, 82, fill=(18, 18, 28))
            cv.circle(ex + g * 80 - 26, 970, 22, fill=(255, 255, 255))
            cv.line([(ex - 190, 760 + o * 8), (ex + 190, 760 - o * 8)], (30, 30, 44), 34)
        cv.rect(-100, 1180, 1200, 1300, fill=dim(FG, 0.92))
    else:  # -------- lamp steady: the champion is back
        lt = t - c_end
        cv.cam = Cam(620, 1040, 1.05 + 0.03 * lt)
        kitchen(cv, t, lamp=1.0, swing=0.8 * math.sin(t), medal_x=400, medal_glint=0.7)
        stools(cv)
        figure(cv, (190, 1205), 560, facing=1, pose=dict(SIT, ra=45, re=50, la=30, le=60), kind="hero", headphones=True, face={"gaze": (1, 0)})
        figure(cv, (890, 1190), 620, facing=-1, pose=dict(SIT, ra=30, re=50, la=25, le=55, head=8), kind="future", face={"gaze": (-1, 0.15)})
        # faint ghost of version B over the champion
        ghost = 0.5 + 0.5 * math.sin(lt * 4)
        if lt < 1.0:
            cv.d.rectangle([0, 0, W, H], fill=(0, 0, 0, 0))
        plate(cv, "2036 · ЧЕМПИОН", YELLOW, lt)
    subtitle(cv, t + sc["start"], sc["lines"])


def run_pose(ph):
    s = math.sin(ph)
    c = math.cos(ph)
    return dict(lean=14, head=-6, lt=-35 * s + 10, lk=-10 - 45 * max(0, c), rt=35 * s + 10, rk=-10 - 45 * max(0, -c),
                la=-40 * s, le=-80, ra=40 * s, re=-80)


def scene_S05(cv, t, sc, fi):
    L0 = sc["lines"][0]
    if t < 2.3:  # -------- dawn kitchen: empty stool with the medal, you take the bag and go
        dawn = 0.25 + 0.5 * seg01(t, 0, 2.3)
        cv.cam = Cam(540, 1050, 0.92)
        kitchen(cv, t, lamp=0.0, swing=0, dawn=dawn, medal_x=None, cup=True, steam=False)
        stools(cv, 0.6 + dawn * 0.4)
        draw_medal(cv, 890, 1205, 24, glint=0.9, squash=0.35)
        # bag on the floor, then in your hand
        stand = ease(seg01(t, 0.2, 0.9))
        walk = seg01(t, 1.0, 2.3)
        hx = 190 + walk * 1200
        if t < 0.9:
            cv.rect(40, 1250, 200, 1330, fill=(30, 36, 80), outline=(70, 80, 140), width=4, r=30)
            pose = dict(SIT) if stand < 0.5 else dict(STAND, la=60 * (1 - stand), le=20)
            hipy = 1205 - 60 * stand
        else:
            pose = run_pose(walk * 14) if walk > 0 else dict(STAND)
            pose.update(la=10, le=5)
            hipy = 1150
        figure(cv, (hx, hipy), 580, facing=1, pose=pose, kind="hero", headphones=True, bag=t >= 0.9, face={"gaze": (1, 0)})
        plate(cv, "2026 · ДЕНЬ 1", GREEN, t)
    elif t < 4.4:  # -------- empty street at dawn, you run, birds take off
        lt = t - 2.3
        cv.cam = Cam(540, 960, 1.0)
        for k in range(20):
            cv.rect(-100, -100 + k * 70, 1200, -30 + k * 70, fill=mix((70, 60, 120), (255, 180, 130), k / 19))
        cv.gcircle(760, 1250, 260, (255, 160, 90), 0.9)
        cv.circle(760, 1290, 120, fill=(255, 214, 160))
        # houses
        random.seed(3)
        x = -100
        while x < 1200:
            w = random.randint(140, 260)
            h = random.randint(260, 600)
            cv.rect(x, 1300 - h, x + w, 1300, fill=(36, 30, 58))
            for wy in range(1300 - h + 40, 1260, 70):
                for wx in range(x + 25, x + w - 30, 60):
                    if random.random() < 0.2:
                        cv.rect(wx, wy, wx + 26, wy + 34, fill=(255, 200, 120), alpha=200)
            x += w + random.randint(0, 20)
        random.seed()
        cv.rect(-100, 1300, 1200, 2100, fill=(30, 26, 44))
        cv.rect(-100, 1300, 1200, 1316, fill=(60, 52, 80))
        # power line with birds
        cv.line([(-100, 560), (300, 600), (700, 590), (1200, 540)], (16, 14, 28), 5)
        cv.line([(300, 600), (300, 1300)], (16, 14, 28), 14, caps=False)
        hx = -150 + lt / 2.1 * 1400
        for i in range(9):
            bx = -40 + i * 130
            by = 590 - (i * 7 % 13)
            take = hx > bx - 250
            if take:
                k = clamp01((hx - (bx - 250)) / 400)
                bx += k * 300 + i * 10 * k
                by -= k * 500 + (i % 3) * 60 * k
                flap = math.sin(lt * 30 + i) * 16
                cv.line([(bx - 22, by - flap), (bx, by), (bx + 22, by - flap)], (16, 14, 28), 6)
            else:
                cv.ellipse(bx, by - 14, 13, 11, fill=(16, 14, 28))
                cv.circle(bx + 9, by - 26, 7, fill=(16, 14, 28))
        figure(cv, (hx, 1035 + abs(math.sin(lt * 9)) * -18), 560, facing=1, pose=run_pose(lt * 9), kind="hero",
               headphones=True, bag=False, rim=(255, 196, 140), face={"gaze": (1, 0)})
        cv.gline([(hx - 300, 1300), (hx + 100, 1300)], (255, 160, 90), 6, 0.4)
    else:  # -------- gym door, low angle; hex lamps light one by one
        lt = t - 4.4
        cv.cam = Cam(540, 960, 1.0 + 0.04 * lt)
        cv.rect(-100, -100, 1200, 2100, fill=(8, 8, 14))
        # ceiling hex lamps seen through the door from below
        lit_n = int(seg01(lt, 0.45, 1.6) * 7)
        pos = [(540, 260), (300, 380), (780, 380), (540, 520), (180, 560), (900, 560), (540, 760)]
        for i, (x, y) in enumerate(pos):
            on = i < lit_n
            cv.poly(hexagon(x, y, 95), outline=(250, 250, 255) if on else (40, 40, 52), width=14)
            if on:
                cv.gpoly(hexagon(x, y, 100), (230, 236, 255), 0.85)
        # door frame and you in it
        open_q = ease(seg01(lt, 0, 0.45))
        cv.poly([(-100, -100), (140 + 0 * open_q, -100), (180, 2100), (-100, 2100)], fill=(26, 26, 36))
        cv.poly([(940 - 300 * (1 - open_q), -100), (1200, -100), (1200, 2100), (900 - 300 * (1 - open_q), 2100)], fill=(40, 38, 54))
        cv.rect(-100, 1700, 1200, 2100, fill=(20, 20, 30))
        figure(cv, (520, 1350), 1150, facing=-1, pose=dict(STAND, ra=-60, re=-20, la=10), kind="hero", headphones=True,
               sil=0.85, rim=(220, 230, 255) if lit_n else (60, 60, 80), number=True)
        cv.wtext(540, 1820, "ЗАЛ · 06:00", 46, MUTED, name="display8")
    subtitle(cv, t + sc["start"], sc["lines"])


def gym(cv, t, sign_on=1.0):
    cv.rect(-800, -800, 2000, 2800, fill=(12, 12, 18))
    # ceiling hex lamps
    for i in range(-3, 6):
        for j in range(2):
            x = i * 260 + (130 if j else 0)
            y = 120 + j * 150
            cv.poly(hexagon(x, y, 70), outline=(235, 238, 255), width=10)
            cv.gpoly(hexagon(x, y, 74), (220, 226, 255), 0.55)
    # mesh wall
    cv.rect(-800, 360, 2000, 1400, fill=(20, 20, 28))
    for k in range(-40, 60):
        cv.line([(k * 40, 360), (k * 40 + 400, 1400)], (38, 38, 50), 3, caps=False)
        cv.line([(k * 40 + 400, 360), (k * 40, 1400)], (38, 38, 50), 3, caps=False)
    # neon sign
    f = 64
    cv.wtext(-250, 520, "ДИСЦИПЛИНА", f, (255, 150, 210), name="display")
    cv.gline([(-470, 520), (-30, 520)], PINK, 70, 0.6 * sign_on)
    # yellow power rack
    for x in (-560, -330):
        cv.line([(x, 640), (x, 1450)], (232, 190, 40), 26, caps=False)
    cv.line([(-580, 660), (-310, 660)], (232, 190, 40), 26, caps=False)
    cv.line([(-640, 940), (-250, 940)], (190, 190, 200), 14)
    cv.circle(-640, 940, 70, outline=(60, 60, 70), width=24)
    cv.circle(-250, 940, 70, outline=(60, 60, 70), width=24)
    # floor + burgundy benches
    cv.rect(-800, 1400, 2000, 2800, fill=(22, 20, 26))
    for bx in (1250, 1600):
        cv.rect(bx - 140, 1300, bx + 140, 1345, fill=(120, 30, 46), r=14)
        cv.rect(bx - 15, 1345, bx + 15, 1420, fill=(40, 40, 50))


def scene_S06(cv, t, sc, fi):
    L0 = sc["lines"][0]
    dur = sc["end"] - sc["start"]
    pull = ease(seg01(t, 0.2, 4.4))
    cv.cam = Cam(560 + 0 * pull, 960 - 40 * pull, 1.55 - 0.85 * pull)
    gym(cv, t)
    # mirror
    mx0, mx1, my0, my1 = 200, 920, 380, 1500
    cv.rect(mx0 - 18, my0 - 18, mx1 + 18, my1 + 18, fill=(70, 70, 84))
    cv.rect(mx0, my0, mx1, my1, fill=(26, 30, 44))
    for k in range(3):
        cv.poly([(mx0 + 80 + k * 200, my0), (mx0 + 150 + k * 200, my0), (mx0 + 20 + k * 200, my1), (mx0 - 50 + k * 200, my1)], fill=(255, 255, 255), alpha=10)
    # reflection: you taking a selfie (seen from the front, slightly dimmer)
    selfie_pose = dict(STAND, ra=110, re=50, la=10, le=10, head=-4)
    figure(cv, (560, 1060), 820, facing=1, pose=selfie_pose, kind="hero", headphones=True, phone=True, phone_glow=0.6,
           face={"gaze": (-0.3, -0.4)}, rim=(170, 176, 200))
    cv.rect(mx0, my0, mx1, my1, fill=(120, 140, 200), alpha=18)
    # camera flash at selfie
    click = 1.3
    if click <= t < click + 0.25:
        q = (t - click) / 0.25
        cv.gcircle(560 + 80, 1060 - 600, 220, (255, 255, 255), 1.4 * (1 - q))
    # honor board with the empty spot
    bx0, by0 = 1000, 420
    cv.rect(bx0, by0, bx0 + 560, by0 + 860, fill=(30, 26, 22), outline=(150, 120, 60), width=10)
    cv.wtext(bx0 + 280, by0 + 70, "ДОСКА ПОЧЁТА", 52, GOLD, name="display8")
    for i in range(3):
        for j in range(2):
            x = bx0 + 60 + j * 240
            y = by0 + 150 + i * 220
            if i == 2 and j == 1:
                for k in range(6):  # dashed empty frame
                    pass
                cv.rect(x, y, x + 200, y + 180, outline=GOLD, width=6, alpha=int(150 + 100 * math.sin(t * 4)))
                cv.wtext(x + 100, y + 92, "?", 110, GOLD)
                cv.grect(x, y, x + 200, y + 180, GOLD, 0.35 * (0.6 + 0.4 * math.sin(t * 4)))
            else:
                cv.rect(x, y, x + 200, y + 180, fill=(60, 56, 70))
                cv.circle(x + 100, y + 70, 36, fill=(120, 116, 130))
                cv.ellipse(x + 100, y + 160, 70, 50, fill=(120, 116, 130))
    # final titles
    tt = t - (L0["end"] - sc["start"]) - 0.1
    if tt > 0:
        a = ease(tt / 0.5)
        cv.d.rectangle([0, 0, W, H], fill=(6, 7, 14, int(150 * a)))
        rows = ["КЕМ ТЫ ХОЧЕШЬ", "БЫТЬ ЧЕРЕЗ", "10 ЛЕТ?"]
        for i, r in enumerate(rows):
            f = font("display", 100 if i < 2 else 180)
            cv.text((W / 2, 700 + i * 150 + (40 if i == 2 else 0) + (1 - a) * 40), r, f, FG if i < 2 else YELLOW,
                    alpha=int(255 * a), stroke=5, stroke_fill=(8, 8, 16))
        a2 = ease((tt - 0.5) / 0.5)
        if a2 > 0:
            f = font("display8", 64)
            txt = "НАПИШИ В КОММЕНТАРИЯХ"
            tw = cv.d.textlength(txt, font=f)
            box = [W / 2 - tw / 2 - 46, 1260, W / 2 + tw / 2 + 46, 1370]
            cv.d.rounded_rectangle(box, radius=26, fill=(12, 14, 26, int(220 * a2)), outline=(*ORANGE, int(255 * a2)), width=6)
            cv.gd.rounded_rectangle([b / GS for b in box], radius=6, outline=dim(ORANGE, a2), width=3)
            cv.text((W / 2, 1316), txt, f, ORANGE, alpha=int(255 * a2))
    else:
        subtitle(cv, t + sc["start"], sc["lines"])


SCENES = {"S01": scene_S01, "S02": scene_S02, "S03": scene_S03, "S04": scene_S04, "S05": scene_S05, "S06": scene_S06}
TLD = TL.build()


def frame_at(fi):
    t = fi / FPS
    for i, sc in enumerate(TLD["scenes"]):
        if sc["start"] <= t < sc["end"] or i == len(TLD["scenes"]) - 1:
            break
    lt = t - sc["start"]
    random.seed(fi * 7919)
    cv = Canvas()
    SCENES[sc["id"]](cv, lt, sc, fi)
    # dip to black around scene boundaries
    fade = 1.0
    if sc["id"] != "S01":
        fade = min(fade, ease(lt / 0.18))
    else:
        fade = min(fade, ease(lt / 0.6))
    rem = sc["end"] - t
    if sc["id"] == "S06":
        fade = min(fade, ease(rem / 0.8))
    else:
        fade = min(fade, 0.15 + 0.85 * ease(rem / 0.14))
    glow_k = 1.0
    if sc["id"] == "S04":
        st = flicker_state(lt) if lt < 3.6 else None
    return finish(cv, fi, glow_k=glow_k, fade=fade)


def main():
    if sys.argv[1] == "--still":
        t = float(sys.argv[2])
        frame_at(int(round(t * FPS))).save(sys.argv[3])
        return
    a, b, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(frame_at(fi).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
