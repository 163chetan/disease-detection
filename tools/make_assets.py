"""
Generates the illustration set used by MedPredict (assets/*.png).
Run from the project root:  python tools/make_assets.py
Pure Pillow, no network needed. Everything is drawn at 2x and downsampled for smooth edges.
"""
import math, os, random
from PIL import Image, ImageDraw, ImageFilter

OUT = "assets"
os.makedirs(OUT, exist_ok=True)
SS = 2  # supersample factor


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(w, h, c1, c2, angle=35):
    """Diagonal linear gradient."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    mx = abs(w * ca) + abs(h * sa)
    for y in range(h):
        for x in range(w):
            t = (x * ca + y * sa) / mx
            px[x, y] = lerp(c1, c2, max(0, min(1, t)))
    return img


def glow(img, cx, cy, r, color, alpha=70):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(r * 0.45))
    img.alpha_composite(layer)


def ecg(d, x0, x1, y, amp, color, width, seed=1):
    """Draw an ECG-like trace."""
    pts, x = [], x0
    beat = amp * 6
    while x < x1:
        pts += [(x, y), (x + beat * .25, y), (x + beat * .32, y - amp * .35), (x + beat * .38, y),
                (x + beat * .45, y + amp * .25), (x + beat * .52, y - amp), (x + beat * .60, y + amp * .55),
                (x + beat * .68, y), (x + beat * .85, y - amp * .3), (x + beat * .95, y)]
        x += beat
    d.line([(min(px, x1), py) for px, py in pts if px <= x1 + 1], fill=color, width=width, joint="curve")


# ───────────────────────── icons (all drawn in white, 0-100 box) ──────────────────────────
W = (255, 255, 255, 235)
WS = (255, 255, 255, 120)


def T(pts, ox, oy, s):
    return [(ox + x * s, oy + y * s) for x, y in pts]


def icon_lungs(d, ox, oy, s):
    lw = max(2, int(3.2 * s))
    # trachea + bronchi
    d.line(T([(50, 8), (50, 40)], ox, oy, s), fill=W, width=int(5 * s), joint="curve")
    d.line(T([(50, 40), (38, 52)], ox, oy, s), fill=W, width=int(4 * s))
    d.line(T([(50, 40), (62, 52)], ox, oy, s), fill=W, width=int(4 * s))
    # lobes
    left = [(44, 30), (30, 34), (14, 56), (10, 80), (16, 92), (30, 92), (42, 84), (44, 60)]
    right = [(56, 30), (70, 34), (86, 56), (90, 80), (84, 92), (70, 92), (58, 84), (56, 60)]
    for lobe in (left, right):
        d.polygon(T(lobe, ox, oy, s), fill=(255, 255, 255, 60))
        d.line(T(lobe + [lobe[0]], ox, oy, s), fill=W, width=lw, joint="curve")
    for a, b in [((34, 56), (24, 70)), ((34, 56), (38, 74)), ((66, 56), (76, 70)), ((66, 56), (62, 74))]:
        d.line(T([a, b], ox, oy, s), fill=WS, width=max(2, int(2.5 * s)))


def icon_heart(d, ox, oy, s):
    pts = []
    for i in range(200):
        t = i / 200 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((50 + x * 2.6, 48 + y * 2.6))
    d.polygon(T(pts, ox, oy, s), fill=(255, 255, 255, 70))
    d.line(T(pts + [pts[0]], ox, oy, s), fill=W, width=int(3.5 * s), joint="curve")
    d.line(T([(18, 52), (36, 52), (42, 38), (50, 70), (58, 44), (63, 52), (82, 52)], ox, oy, s),
           fill=W, width=int(3 * s), joint="curve")


def icon_ribbon(d, ox, oy, s):
    lw = int(7 * s)
    d.arc([ox + 30 * s, oy + 8 * s, ox + 70 * s, oy + 56 * s], 0, 360, fill=W, width=lw)
    d.line(T([(40, 50), (24, 92)], ox, oy, s), fill=W, width=lw, joint="curve")
    d.line(T([(60, 50), (76, 92)], ox, oy, s), fill=W, width=lw, joint="curve")
    d.line(T([(44, 44), (70, 80)], ox, oy, s), fill=W, width=lw)
    d.line(T([(56, 44), (30, 80)], ox, oy, s), fill=W, width=lw)


def icon_brain(d, ox, oy, s):
    blobs = [(38, 38, 20), (62, 38, 20), (30, 58, 18), (70, 58, 18), (50, 70, 18), (50, 48, 20)]
    for cx, cy, r in blobs:
        d.ellipse([ox + (cx - r) * s, oy + (cy - r) * s, ox + (cx + r) * s, oy + (cy + r) * s],
                  fill=(255, 255, 255, 38), outline=W, width=max(2, int(2.6 * s)))
    d.line(T([(50, 30), (50, 86)], ox, oy, s), fill=WS, width=int(2.4 * s))
    d.line(T([(34, 44), (46, 52)], ox, oy, s), fill=WS, width=int(2.4 * s))
    d.line(T([(66, 44), (54, 52)], ox, oy, s), fill=WS, width=int(2.4 * s))


def icon_liver(d, ox, oy, s):
    pts = [(8, 50), (20, 32), (44, 26), (70, 30), (90, 44), (94, 58), (82, 74), (56, 84), (34, 78), (16, 66)]
    d.polygon(T(pts, ox, oy, s), fill=(255, 255, 255, 70))
    d.line(T(pts + [pts[0]], ox, oy, s), fill=W, width=int(3.5 * s), joint="curve")
    d.line(T([(58, 28), (60, 56), (52, 80)], ox, oy, s), fill=WS, width=int(3 * s), joint="curve")
    d.ellipse([ox + 63 * s, oy + 54 * s, ox + 77 * s, oy + 68 * s], outline=W, width=int(2.5 * s))


def icon_kidney(d, ox, oy, s):
    for flip in (1, -1):
        pts = []
        for i in range(0, 360, 6):
            a = math.radians(i)
            notch = 1 - 0.42 * math.exp(-((((i - (180 if flip == 1 else 0)) + 180) % 360 - 180) / 38.0) ** 2)
            x = 19 * math.cos(a) * notch
            y = 31 * math.sin(a) * notch
            pts.append((50 + flip * 21 + x * (1 if flip == 1 else 1), 52 + y))
        d.polygon(T(pts, ox, oy, s), fill=(255, 255, 255, 70))
        d.line(T(pts + [pts[0]], ox, oy, s), fill=W, width=int(3.5 * s), joint="curve")
    d.line(T([(50, 52), (50, 10)], ox, oy, s), fill=WS, width=int(3 * s))
    d.line(T([(50, 10), (36, 10)], ox, oy, s), fill=WS, width=int(3 * s))


def icon_drop(d, ox, oy, s):
    pts = [(50, 6)]
    for i in range(-40, 221, 5):
        a = math.radians(i)
        pts.append((50 + 31 * math.cos(a), 64 + 31 * math.sin(a)))
    d.polygon(T(pts, ox, oy, s), fill=(255, 255, 255, 70))
    d.line(T(pts + [pts[0]], ox, oy, s), fill=W, width=int(3.5 * s), joint="curve")
    d.arc([ox + 34 * s, oy + 48 * s, ox + 68 * s, oy + 82 * s], 95, 185, fill=WS, width=int(3.5 * s))


def icon_virus(d, ox, oy, s):
    cx, cy = 50, 50
    for i in range(12):
        a = math.radians(i * 30)
        x1, y1 = cx + 26 * math.cos(a), cy + 26 * math.sin(a)
        x2, y2 = cx + 40 * math.cos(a), cy + 40 * math.sin(a)
        d.line(T([(x1, y1), (x2, y2)], ox, oy, s), fill=W, width=int(3 * s))
        d.ellipse([ox + (x2 - 4) * s, oy + (y2 - 4) * s, ox + (x2 + 4) * s, oy + (y2 + 4) * s], fill=W)
    d.ellipse([ox + 24 * s, oy + 24 * s, ox + 76 * s, oy + 76 * s], fill=(255, 255, 255, 70), outline=W, width=int(3.5 * s))
    for x, y in [(42, 44), (58, 52), (46, 60)]:
        d.ellipse([ox + (x - 4) * s, oy + (y - 4) * s, ox + (x + 4) * s, oy + (y + 4) * s], fill=WS)


def icon_search(d, ox, oy, s):
    d.ellipse([ox + 12 * s, oy + 10 * s, ox + 68 * s, oy + 66 * s], fill=(255, 255, 255, 50), outline=W, width=int(5 * s))
    d.line(T([(60, 58), (90, 90)], ox, oy, s), fill=W, width=int(8 * s))
    d.line(T([(40, 24), (40, 52)], ox, oy, s), fill=W, width=int(5 * s))
    d.line(T([(26, 38), (54, 38)], ox, oy, s), fill=W, width=int(5 * s))


def icon_shield(d, ox, oy, s, mark):
    pts = [(50, 6), (86, 20), (86, 50), (50, 94), (14, 50), (14, 20)]
    d.polygon(T(pts, ox, oy, s), fill=(255, 255, 255, 60))
    d.line(T(pts + [pts[0]], ox, oy, s), fill=W, width=int(4 * s), joint="curve")
    if mark == "ok":
        d.line(T([(32, 48), (46, 62), (70, 34)], ox, oy, s), fill=W, width=int(7 * s), joint="curve")
    else:
        d.line(T([(50, 26), (50, 54)], ox, oy, s), fill=W, width=int(7 * s))
        d.ellipse([ox + 44 * s, oy + 62 * s, ox + 56 * s, oy + 74 * s], fill=W)


ICONS = {"lungs": icon_lungs, "heart": icon_heart, "ribbon": icon_ribbon, "brain": icon_brain,
         "liver": icon_liver, "kidney": icon_kidney, "drop": icon_drop, "virus": icon_virus,
         "search": icon_search}


class Layer:
    """Transparent drawing layer; flush() alpha-composites it onto the base image so alpha really blends."""
    def __init__(self, img):
        self.img = img
        self.layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)

    def flush(self):
        self.img.alpha_composite(self.layer)
        self.layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)


def decorate(img, seed):
    rnd = random.Random(seed)
    w, h = img.size
    for _ in range(3):
        glow(img, rnd.randint(0, w), rnd.randint(0, h), rnd.randint(h // 3, int(h // 1.5)), (255, 255, 255), 28)
    L = Layer(img)
    d = L.d
    for _ in range(40):  # bokeh dots
        x, y, r = rnd.randint(0, w), rnd.randint(0, h), rnd.randint(2, 7) * SS
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, rnd.randint(18, 45)))
    for _ in range(9):  # plus signs
        x, y, r = rnd.randint(0, w), rnd.randint(0, h), rnd.randint(8, 16) * SS
        c = (255, 255, 255, rnd.randint(25, 55))
        d.line([(x - r, y), (x + r, y)], fill=c, width=3 * SS)
        d.line([(x, y - r), (x, y + r)], fill=c, width=3 * SS)
    L.flush()


def banner(name, c1, c2, icon, w=1400, h=360, seed=1):
    W2, H2 = w * SS, h * SS
    img = gradient(W2, H2, c1, c2).convert("RGBA")
    decorate(img, seed)
    L = Layer(img)
    ecg(L.d, 0, W2, int(H2 * .78), 26 * SS, (255, 255, 255, 60), 3 * SS)
    L.flush()
    size = int(H2 * .86)
    cx, cy = int(W2 * .82), int(H2 * .5)
    glow(img, cx, cy, int(size * .75), (255, 255, 255), 55)
    L = Layer(img)
    d = L.d
    d.ellipse([cx - size * .62, cy - size * .62, cx + size * .62, cy + size * .62], outline=(255, 255, 255, 70), width=3 * SS)
    d.ellipse([cx - size * .52, cy - size * .52, cx + size * .52, cy + size * .52], fill=(255, 255, 255, 28))
    ICONS[icon](d, cx - size * .38, cy - size * .38, size * .76 / 100)
    L.flush()
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(f"{OUT}/{name}.jpg", quality=88, optimize=True)


def hero(w=1600, h=620):
    W2, H2 = w * SS, h * SS
    img = gradient(W2, H2, (14, 32, 84), (14, 116, 144), angle=20).convert("RGBA")
    decorate(img, 7)
    glow(img, int(W2 * .78), int(H2 * .45), int(H2 * .6), (45, 212, 191), 70)
    glow(img, int(W2 * .15), int(H2 * .9), int(H2 * .5), (99, 102, 241), 60)
    L = Layer(img)
    d = L.d
    ecg(d, 0, W2, int(H2 * .86), 34 * SS, (45, 212, 191, 130), 4 * SS)
    ecg(d, int(W2 * .06), W2, int(H2 * .90), 22 * SS, (255, 255, 255, 60), 3 * SS)
    tiles = [("lungs", .60, .10, .36), ("heart", .78, .28, .38), ("brain", .58, .50, .30),
             ("drop", .90, .06, .26), ("ribbon", .73, .66, .22)]
    for ic, fx, fy, fs in tiles:
        sz = int(H2 * fs)
        x, y = int(W2 * fx), int(H2 * fy)
        d.rounded_rectangle([x, y, x + sz, y + sz], radius=int(sz * .24), fill=(255, 255, 255, 30), outline=(255, 255, 255, 90), width=2 * SS)
        ICONS[ic](d, x + sz * .16, y + sz * .16, sz * .68 / 100)
    L.flush()
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(f"{OUT}/hero.jpg", quality=88, optimize=True)


def result_img(name, c1, c2, mark, w=720, h=260):
    W2, H2 = w * SS, h * SS
    img = gradient(W2, H2, c1, c2, angle=15).convert("RGBA")
    decorate(img, 11 if mark == "ok" else 12)
    L = Layer(img)
    ecg(L.d, 0, W2, int(H2 * .5), 30 * SS, (255, 255, 255, 70), 3 * SS)
    L.flush()
    size = int(H2 * .82)
    cx, cy = W2 // 2, H2 // 2
    glow(img, cx, cy, int(size * .7), (255, 255, 255), 70)
    L = Layer(img)
    icon_shield(L.d, cx - size * .4, cy - size * .42, size * .8 / 100, mark)
    L.flush()
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(f"{OUT}/{name}.jpg", quality=88, optimize=True)


def logo(w=512):
    S2 = w * SS
    base = Image.new("RGBA", (S2, S2), (0, 0, 0, 0))
    g = gradient(S2, S2, (37, 99, 235), (13, 148, 136), angle=45).convert("RGBA")
    mask = Image.new("L", (S2, S2), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S2, S2], radius=int(S2 * .24), fill=255)
    base.paste(g, (0, 0), mask)
    glow(base, int(S2 * .3), int(S2 * .25), int(S2 * .35), (255, 255, 255), 50)
    # clip glow back to the rounded square
    clipped = Image.new("RGBA", (S2, S2), (0, 0, 0, 0))
    clipped.paste(base, (0, 0), mask)
    base = clipped
    L = Layer(base)
    d = L.d
    a, t = int(S2 * .2), int(S2 * .17)
    cx = cy = S2 // 2
    d.rounded_rectangle([cx - t // 2, cy - a - t // 2, cx + t // 2, cy + a + t // 2], radius=t // 3, fill=(255, 255, 255, 245))
    d.rounded_rectangle([cx - a - t // 2, cy - t // 2, cx + a + t // 2, cy + t // 2], radius=t // 3, fill=(255, 255, 255, 245))
    ecg(d, int(S2 * .14), int(S2 * .86), int(S2 * .74), int(S2 * .055), (20, 184, 166, 255), int(S2 * .022))
    L.flush()
    base.resize((w, w), Image.LANCZOS).save(f"{OUT}/logo.png", optimize=True)


def login_art(w=900, h=1100):
    W2, H2 = w * SS, h * SS
    img = gradient(W2, H2, (12, 28, 78), (13, 120, 140), angle=70).convert("RGBA")
    decorate(img, 21)
    glow(img, int(W2 * .5), int(H2 * .3), int(W2 * .5), (45, 212, 191), 60)
    L = Layer(img)
    d = L.d
    ecg(d, 0, W2, int(H2 * .62), 42 * SS, (255, 255, 255, 90), 4 * SS)
    sz = int(W2 * .5)
    cx, cy = W2 // 2, int(H2 * .3)
    d.ellipse([cx - sz * .62, cy - sz * .62, cx + sz * .62, cy + sz * .62], fill=(255, 255, 255, 26), outline=(255, 255, 255, 80), width=3 * SS)
    icon_heart(d, cx - sz * .4, cy - sz * .4, sz * .8 / 100)
    L.flush()
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(f"{OUT}/login_art.jpg", quality=88, optimize=True)


PALETTE = {
    "general":   ((30, 64, 175), (14, 165, 233), "search"),
    "diabetes":  ((29, 78, 216), (56, 189, 248), "drop"),
    "heart":     ((190, 24, 93), (251, 113, 133), "heart"),
    "parkinson": ((91, 33, 182), (167, 139, 250), "brain"),
    "liver":     ((180, 83, 9), (251, 191, 36), "liver"),
    "hepatitis": ((154, 52, 18), (251, 146, 60), "virus"),
    "lung":      ((3, 105, 161), (45, 212, 191), "lungs"),
    "kidney":    ((15, 118, 110), (110, 231, 183), "kidney"),
    "breast":    ((157, 23, 77), (244, 114, 182), "ribbon"),
}

if __name__ == "__main__":
    hero()
    logo()
    login_art()
    result_img("result_positive", (159, 18, 57), (249, 115, 22), "alert")
    result_img("result_negative", (4, 120, 87), (45, 212, 191), "ok")
    for i, (k, (c1, c2, ic)) in enumerate(PALETTE.items()):
        banner(k, c1, c2, ic, seed=i + 3)
    print("done:", sorted(os.listdir(OUT)))
