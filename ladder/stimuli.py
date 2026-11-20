"""Stimuli: 25 illusions x 3 conditions x 4 renders = 300 PNGs.

Every image is drawn here from parameters, at 2x and downsampled for
anti-aliasing. Conditions:
  normal  - the illusion
  reverse - the inducing element reversed. For two-sided figures (which line is longer, ...) this moves the
            illusion to the other side; for yes/no questions it is the other form of the illusion (Wundt for
            Hering, opposite twist, opposite polarity), so the illusion answer stays the same word.
  none    - the inducing element removed. In illusions marked CHECK, renders 2 and 3 of `none` get a real
            difference (first / second item larger by 10 %); their correct answer is that item.
Renders within a condition vary colours, size, line width, position, and (r odd) orientation; they never move
the illusion to the other side, so normal vs reverse is the position counterbalance for two-sided figures.

  python3 -m ladder.stimuli [out_dir]     writes the PNGs, manifest.csv, manifest.json and contact sheets
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 900
SS = 2
OUT = Path(__file__).resolve().parent.parent / "stimuli"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CONDS = ("normal", "reverse", "none")
RENDERS = 4
D = 0.10
# line colour / background per render (for line drawings)
SCHEMES = [((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0)),
           ((25, 35, 95), (245, 240, 225)), ((60, 60, 60), (205, 205, 205))]


class Canvas:
    """Drawing in local coordinates centred on (0, 0); the canvas scales, turns and moves the figure.

    turn: '' | 'T' (transpose x<->y) | 'R90' | 'R180' | 'FY' (flip vertically). Local designs stay within
    x +-540, y +-400; turned designs (T, R90) are shrunk so they fit the 900 px height, unless shrink=False
    (for square designs that fit either way).
    """

    def __init__(self, bg, rng, turn="", mode="RGB", scale=None, shrink=True):
        self.turn = turn
        s = scale if scale is not None else rng.uniform(0.92, 1.04)
        if turn in ("T", "R90") and shrink:
            s *= 0.74
        self.s = s
        self.cx = W / 2 + rng.uniform(-15, 15)
        self.cy = H / 2 + rng.uniform(-12, 12)
        self.im = Image.new(mode, (W * SS, H * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    def p(self, x, y):
        t = self.turn
        if t == "T":
            x, y = y, x
        elif t == "R90":
            x, y = -y, x
        elif t == "R180":
            x, y = -x, -y
        elif t == "FY":
            y = -y
        return ((self.cx + self.s * x) * SS, (self.cy + self.s * y) * SS)

    def k(self, v):
        return v * self.s * SS

    def line(self, pts, fill, width):
        self.d.line([self.p(*q) for q in pts], fill=fill, width=max(1, round(self.k(width))), joint="curve")

    def poly(self, pts, fill, outline=None, width=0):
        self.d.polygon([self.p(*q) for q in pts], fill=fill)
        if outline is not None and width:
            q = list(pts) + [pts[0]]
            self.line(q, outline, width)

    def rect(self, x0, y0, x1, y1, fill):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill)

    def circle(self, x, y, r, fill=None, outline=None, width=0):
        cx, cy = self.p(x, y)
        rr = self.k(r)
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=fill, outline=outline,
                       width=max(1, round(self.k(width))) if outline is not None else 0)

    def sector(self, x, y, r0, r1, a0, a1, fill, steps=None):
        """Annular sector between radii r0 < r1 and angles a0..a1 (radians, image convention: +y down)."""
        steps = steps or max(3, int(abs(a1 - a0) * r1 / 6))
        outer = [(x + r1 * math.cos(a0 + (a1 - a0) * i / steps), y + r1 * math.sin(a0 + (a1 - a0) * i / steps))
                 for i in range(steps + 1)]
        inner = [(x + r0 * math.cos(a1 - (a1 - a0) * i / steps), y + r0 * math.sin(a1 - (a1 - a0) * i / steps))
                 for i in range(steps + 1)]
        self.poly(outer + inner, fill)

    def text(self, x, y, s, fill, size):
        f = ImageFont.truetype(FONT, max(8, round(self.k(size))))
        self.d.text(self.p(x, y), s, fill=fill, font=f, anchor="mm")

    def done(self):
        return self.im.resize((W, H), Image.LANCZOS)


def pair(d):
    """Size factors of (first, second) for a real difference d (+ = first larger)."""
    return 1 + max(d, 0), 1 + max(-d, 0)


def side_words(turn):
    return ("left", "right") if turn == "T" else ("top", "bottom")


def three(first, second, same, ill, d):
    """meta for a two-sided figure. ill = 0 (first), 1 (second) or None; d = real difference."""
    opts = [first, second, same]
    correct = same if d == 0 else (first if d > 0 else second)
    return {"options": opts, "correct": correct, "illusion_answer": None if ill is None else opts[ill]}


def yesno(illusion_yes):
    """meta for a yes/no question; illusion_yes says whether the illusion pushes towards 'yes'."""
    return {"options": ["yes", "no"], "correct": "no" if illusion_yes else "yes",
            "illusion_answer": "yes" if illusion_yes else "no"}


def lw(rng, base):
    return base * rng.uniform(0.8, 1.2)


# ---------------------------------------------------------------- G: size, length, angle, shape

def muller_lyer(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "T" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    L, f, a, w = 400, rng.uniform(60, 80), math.radians(rng.uniform(30, 40)), lw(rng, 6)
    k1, k2 = pair(d)
    first_out = cond == "normal"
    for y, k, out in ((-130, k1, first_out), (130, k2, not first_out)):
        x0, x1 = -L * k / 2, L * k / 2
        c.line([(x0, y), (x1, y)], fg, w)
        if cond == "none":
            continue
        dx, dy = f * math.cos(a), f * math.sin(a)
        sx = 1 if out else -1           # out: fins open away from the line (>--<), looks longer
        for x, side in ((x1, 1), (x0, -1)):
            for sy in (1, -1):
                c.line([(x, y), (x + side * sx * dx, y + sy * dy)], fg, w)
    ill = None if cond == "none" else (0 if first_out else 1)
    return c.done(), three(*side_words(turn), "same", ill, d), {"fin": round(f), "angle": round(math.degrees(a))}


def ponzo(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "T" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    w = lw(rng, 6)
    narrow, wide = 50, 330
    if cond == "none":
        top, bot = 200, 200
    elif cond == "normal":
        top, bot = narrow, wide          # rails meet at the top: the top bar looks longer
    else:
        top, bot = wide, narrow
    for sx in (-1, 1):
        c.line([(sx * bot, 360), (sx * top, -360)], fg, w)
    half = 80
    k1, k2 = pair(d)
    for y, k in ((-170, k1), (170, k2)):
        c.line([(-half * k, y), (half * k, y)], fg, w * 1.6)
    ill = {"normal": 0, "reverse": 1}.get(cond)
    return c.done(), three(*side_words(turn), "same", ill, d), {}


def ebbinghaus(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "T" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    R = 50
    k1, k2 = pair(d)
    small = (8, 18, 92)                 # count, radius, distance: makes the centre look larger
    large = (5, 72, 172)
    phase = rng.uniform(0, 2 * math.pi)
    surround = {"normal": (small, large), "reverse": (large, small), "none": (None, None)}[cond]
    for x, k, sur in ((-255, k1, surround[0]), (255, k2, surround[1])):
        c.circle(x, 0, R * k, fill=fg)
        if sur:
            n, rr, dist = sur
            for i in range(n):
                t = phase + 2 * math.pi * i / n
                c.circle(x + dist * math.cos(t), dist * math.sin(t), rr, fill=fg)
    first, second = ("top", "bottom") if turn == "T" else ("left", "right")
    ill = {"normal": 0, "reverse": 1}.get(cond)
    return c.done(), three(first, second, "same", ill, d), {}


def delboeuf(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "T" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    R, w = 62, lw(rng, 5)
    k1, k2 = pair(d)
    near, far = 80, 160                 # ring close to the disc: disc looks larger
    rings = {"normal": (near, far), "reverse": (far, near), "none": (None, None)}[cond]
    for x, k, ring in ((-250, k1, rings[0]), (250, k2, rings[1])):
        c.circle(x, 0, R * k, fill=fg)
        if ring:
            c.circle(x, 0, ring, outline=fg, width=w)
    first, second = ("top", "bottom") if turn == "T" else ("left", "right")
    ill = {"normal": 0, "reverse": 1}.get(cond)
    return c.done(), three(first, second, "same", ill, d), {}


def vertical_horizontal(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "R180" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    L, w = 400, lw(rng, 7)
    kv, kh = pair(d)                    # first = vertical
    if cond == "normal":                # inverted T: the vertical bisects the horizontal and looks longer
        c.line([(-L / 2, 200), (L / 2, 200)], fg, w)
        c.line([(0, 200), (0, -200)], fg, w)
        ill = 0
    elif cond == "reverse":             # T on its side: the horizontal bisects the vertical (bisection effect)
        c.line([(-200, -L / 2), (-200, L / 2)], fg, w)
        c.line([(-200, 0), (200, 0)], fg, w)
        ill = 1
    else:                               # separate lines, no bisection
        c.line([(-260, -L * kv / 2), (-260, L * kv / 2)], fg, w)
        c.line([(-120, 0), (-120 + L * kh, 0)], fg, w)
        ill = None
    return c.done(), three("vertical", "horizontal", "same", ill, d), {}


def jastrow(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    turn = "T" if r % 2 else ""
    c = Canvas(bg, rng, turn)
    k1, k2 = pair(d)
    if cond == "none":
        for y, k in ((-90, k1), (90, k2)):
            c.rect(-165 * k, y - 60, 165 * k, y + 60, fg)
        return c.done(), three(*side_words(turn), "same", None, d), {}
    R1, R2, span = 330, 200, math.radians(60)
    flip = 1 if cond == "normal" else -1         # normal: arches (convex up), the lower one looks larger
    a0, a1 = -math.pi / 2 - span / 2, -math.pi / 2 + span / 2
    for cy in (176, 326):
        outer = [(R1 * math.cos(a0 + (a1 - a0) * i / 40), cy + R1 * math.sin(a0 + (a1 - a0) * i / 40)) for i in range(41)]
        inner = [(R2 * math.cos(a1 - (a1 - a0) * i / 40), cy + R2 * math.sin(a1 - (a1 - a0) * i / 40)) for i in range(41)]
        c.poly([(x, flip * y) for x, y in outer + inner], fg)
    return c.done(), three(*side_words(turn), "same", 1 if cond == "normal" else 0, d), {}


def poggendorff(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    c = Canvas(bg, rng, "")
    w = lw(rng, 6)
    half_bar = rng.uniform(84, 102)                       # 1.2x the first version (user, 2026-10-13)
    arm = rng.uniform(190, 230)                           # length of each oblique part, measured along x
    theta = math.radians(0 if cond == "none" else 45)
    slope = math.tan(theta) * (1 if cond == "normal" else -1)   # normal: rising to the right
    shift = {2: -28, 3: 28}.get(r, 0) if cond == "none" else 0  # image y: - = higher
    y_at = lambda x: -slope * x
    x0, x1 = half_bar + arm, half_bar
    c.line([(-x0, y_at(-x0)), (-x1, y_at(-x1))], fg, w)
    c.line([(x1, y_at(x1) + shift), (x0, y_at(x0) + shift)], fg, w)
    c.rect(-half_bar, -380, half_bar, 380, fg)
    # the right part of a rising line looks too high (the acute angles look larger), of a falling line too low
    ill = {"normal": 0, "reverse": 1}.get(cond)
    meta = {"options": ["higher", "lower", "aligned"],
            "correct": "aligned" if shift == 0 else ("higher" if shift < 0 else "lower"),
            "illusion_answer": None if ill is None else ["higher", "lower"][ill]}
    return c.done(), meta, {"shift_px": shift, "theta": round(math.degrees(theta), 1), "bar": round(2 * half_bar)}


def zollner(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    c = Canvas(bg, rng, "R90" if r % 2 else "")
    w = lw(rng, 5)
    ang = math.radians(rng.uniform(40, 50))
    hl, step = 34, 48
    for i, y in enumerate(range(-250, 251, 100)):
        c.line([(-460, y), (460, y)], fg, w)
        if cond == "none":
            a = math.pi / 2
        else:
            a = ang if (i % 2 == 0) == (cond == "normal") else -ang
        for x in range(-440, 441, step):
            dx, dy = hl * math.cos(a), hl * math.sin(a)
            c.line([(x - dx, y - dy), (x + dx, y + dy)], fg, w)
    return c.done(), yesno(False), {}


def cafe_wall(cond, r, rng, d):
    c = Canvas((255, 255, 255), rng, "R90" if r % 2 else "")
    T = rng.uniform(70, 86)
    rows, mortar = 7, rng.uniform(5, 7)
    off = {"normal": 0.5, "reverse": -0.5, "none": 1.0}[cond]
    top = -rows * T / 2
    c.rect(-480, top, 480, top + rows * T, (255, 255, 255))
    for k in range(rows):
        y0 = top + k * T
        x = -480 - 2 * T + (k % 2) * off * T
        n = 0
        while x < 480:
            if n % 2 == 0 and x + T > -480:
                c.rect(max(x, -480), y0, min(x + T, 480), y0 + T, (0, 0, 0))
            x += T
            n += 1
    for k in range(rows + 1):
        y = top + k * T
        c.line([(-480, y), (480, y)], (128, 128, 128), mortar)
    return c.done(), yesno(False), {"tile": round(T)}


def hering(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    c = Canvas(bg, rng, "R90" if r % 2 else "")
    w, wt = lw(rng, 3), lw(rng, 8)
    ray = (210, 40, 40) if r != 3 else (150, 20, 20)
    if cond == "normal":                # rays from the centre: the lines bow outwards
        for i in range(32):
            t = math.pi * i / 32
            dx, dy = math.cos(t), math.sin(t)
            for sg in (1, -1):
                c.line([(sg * 30 * dx, sg * 30 * dy), (sg * 700 * dx, sg * 700 * dy)], ray, w)
    elif cond == "reverse":             # rays from points outside the lines (Wundt): the lines bow inwards
        for cy, sgn in ((-420, 1), (420, -1)):
            for i in range(1, 30):
                t = math.pi * i / 30
                c.line([(0, cy), (700 * math.cos(t), cy + sgn * 700 * math.sin(t))], ray, w)
    else:                               # crossings at right angles: no bending
        for x in range(-480, 481, 40):
            c.line([(x, -380), (x, 380)], ray, w)
    for y in (-110, 110):
        c.line([(-500, y), (500, y)], fg, wt)
    return c.done(), yesno(False), {}


def fraser_spiral(cond, r, rng, d):
    """After Fraser (1908) as usually reproduced: a background of tilted diamond checks (grey with black and white)
    and concentric 'twisted cords', each a chain of short tilted pieces made of a black and a white strand whose
    order alternates from piece to piece. Drawn in log-polar coordinates (u = ln r, v = angle), where circles are
    horizontal lines and the tilted elements are straight, then sampled per pixel at 2x."""
    sgn = {"normal": 1, "reverse": -1, "none": 0}[cond]
    N = rng.choice([36, 40])                     # cells around the circle
    alpha = math.radians(rng.uniform(16, 20)) * sgn   # tilt of the cord pieces against the circle
    delta = math.radians(14) * sgn               # tilt of the background diamonds
    rot = rng.uniform(0, 2 * math.pi)
    scale = rng.uniform(0.95, 1.05)
    cx, cy = W / 2 + rng.uniform(-15, 15), H / 2 + rng.uniform(-12, 12)
    k = N / (2 * math.pi)                        # conformal: one cell is square in (u, v)
    cord_every, half_w = 2.0, 0.24               # a cord every 2 cells in u; half width in cells
    r_out = 430 * scale
    ca, sa = math.cos(math.pi / 4 + delta), math.sin(math.pi / 4 + delta)
    cb, sb = math.cos(math.pi / 4 - delta), math.sin(math.pi / 4 - delta)
    ta = math.tan(alpha)
    GREY, BLACK, WHITE = 150, 0, 255
    img = Image.new("L", (W * SS, H * SS), 128)
    px = []
    for yy in range(H * SS):
        y = yy / SS + 0.5 / SS - cy
        row = []
        for xx in range(W * SS):
            x = xx / SS + 0.5 / SS - cx
            rr = math.hypot(x, y)
            if rr > r_out or rr < 55 * scale:
                row.append(128)
                continue
            u = k * math.log(rr / r_out)
            v = k * (math.atan2(y, x) + rot)
            # cord: pieces of length 1 cell in v, each tilted by alpha; two strands (outer / inner)
            uc = round(u / cord_every) * cord_every
            s_i = math.floor(v)
            centre = uc + ta * (v - s_i - 0.5)
            du = u - centre
            if abs(du) < half_w:
                outer = du > 0
                row.append(BLACK if outer == (s_i % 2 == 0) else WHITE)
                continue
            # background: diamonds from two families of tilted lines
            a_ = math.floor((u * ca + v * sa) * 1.4142)
            b_ = math.floor((-u * cb + v * sb) * 1.4142)
            if (a_ + b_) % 2 == 0:
                row.append(GREY)
            else:
                row.append(BLACK if a_ % 2 == 0 else WHITE)
        px.extend(row)
    img.putdata(px)
    out = img.resize((W, H), Image.LANCZOS).convert("RGB")
    return out, yesno(True), {"cells": N, "alpha": round(math.degrees(alpha), 1)}


# ---------------------------------------------------------------- L: lightness

def simultaneous_contrast(cond, r, rng, d):
    turn = "T" if r % 2 else ""
    c = Canvas((255, 255, 255), rng, turn)
    grey = rng.randint(118, 138)
    dark, light = rng.randint(15, 45), rng.randint(210, 240)
    bgs = {"normal": (dark, light), "reverse": (light, dark), "none": (200, 200)}[cond]
    k1, k2 = pair(d)
    for x, b, k in ((-230, bgs[0], k1), (230, bgs[1], k2)):
        c.rect(x - 220, -220, x + 220, 220, (b,) * 3)
        g = min(255, round(grey * k))
        c.rect(x - 70, -70, x + 70, 70, (g,) * 3)
    first, second = ("top", "bottom") if turn == "T" else ("left", "right")
    ill = {"normal": 0, "reverse": 1}.get(cond)   # the grey on the dark ground looks lighter
    return c.done(), three(first, second, "same", ill, d), {"grey": grey, "dark": dark, "light": light}


def checker_shadow(cond, r, rng, d):
    turn = ["", "R90", "R180", "FY"][r]
    sc = rng.uniform(0.92, 1.04)
    n, T = 7, 92
    light, dark = 200, 100
    c = Canvas((255, 255, 255), rng, turn, scale=sc, shrink=False)   # the board is square: same size in every turn
    half = n * T / 2
    for i in range(n):
        for j in range(n):
            col = light if (i + j) % 2 == 0 else dark
            c.rect(-half + j * T, -half + i * T, -half + (j + 1) * T, -half + (i + 1) * T, (col,) * 3)
    # shadow centred on the light square (i, j) = (4, 4); A = dark square (1, 2) outside it
    bi, bj, ai, aj = 4, 4, 1, 2
    sq = lambda i, j: (-half + j * T, -half + i * T, -half + (j + 1) * T, -half + (i + 1) * T)
    if cond != "none":
        m = Canvas((255, 255, 255), random.Random(0), turn, scale=sc, shrink=False)
        m.cx, m.cy = c.cx, c.cy
        bx, by = -half + (bj + 0.5) * T, -half + (bi + 0.5) * T
        m.circle(bx, by, 1.9 * T, fill=(128, 128, 128))
        mask = m.im.filter(ImageFilter.GaussianBlur(radius=c.k(0.35 * T)))
        c.im = ImageChops.multiply(c.im, mask)
        c.d = ImageDraw.Draw(c.im)
        g1 = g2 = round(light * 128 / 255)          # = the lit dark square, made exact
        c.rect(*sq(bi, bj), (g2,) * 3)
        c.rect(*sq(ai, aj), (g1,) * 3)
        # normal: B is the square in the shadow (looks lighter); reverse: the labels are swapped
        lab_shadow, lab_lit = ("B", "A") if cond == "normal" else ("A", "B")
        ill = lab_shadow
    else:
        g = 150
        k1, k2 = pair(d)                            # first = A
        lab_lit = None
        c.rect(*sq(ai, aj), (min(255, round(g * k1)),) * 3)
        c.rect(*sq(bi - 1, bj), (min(255, round(g * k2)),) * 3)       # both on dark-square positions, ringed by light squares
        ill = None
    out = c.done()
    dd = ImageDraw.Draw(out)
    f = ImageFont.truetype(FONT, round(30 * c.s))
    def label(i, j, s):
        x0, y0 = c.p(-half + (j + 0.5) * T, -half + (i + 0.5) * T)
        dd.text((x0 / SS, y0 / SS), s, font=f, fill=(255, 40, 40), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    if cond == "none":
        label(ai, aj, "A")
        label(bi - 1, bj, "B")
        meta = three("A", "B", "same", None, d)
    else:
        label(bi, bj, lab_shadow)
        label(ai, aj, lab_lit)
        meta = {"options": ["A", "B", "same"], "correct": "same", "illusion_answer": ill}
    return out, meta, {}


def white_effect(cond, r, rng, d):
    turn = "T" if r % 2 else ""
    c = Canvas((255, 255, 255), rng, turn)
    s = rng.uniform(34, 42)
    rows = 12
    top = -rows * s / 2
    grey = rng.randint(118, 138)
    k1, k2 = pair(d)
    if cond == "none":
        c.rect(-440, top, 440, -top, (190, 190, 190))
    else:
        for i in range(rows):
            col = (0, 0, 0) if i % 2 else (255, 255, 255)
            c.rect(-440, top + i * s, 440, top + (i + 1) * s, col)
    # normal: the left greys sit on black stripes (look lighter); reverse: on white stripes
    left_rows = [i for i in range(2, rows - 2) if (i % 2 == 1) == (cond != "reverse")][:3]
    right_rows = [i for i in range(2, rows - 2) if (i % 2 == 1) == (cond == "reverse")][:3]
    for rows_, x0, k in ((left_rows, -330, k1), (right_rows, 130, k2)):
        g = min(255, round(grey * k))
        for i in rows_:
            c.rect(x0, top + i * s, x0 + 200, top + (i + 1) * s, (g,) * 3)
    first, second = ("top", "bottom") if turn == "T" else ("left", "right")
    ill = {"normal": 0, "reverse": 1}.get(cond)
    return c.done(), three(first, second, "same", ill, d), {"grey": grey}


def hermann_grid(cond, r, rng, d):
    pol = (cond == "reverse")
    sq_col, st_col = ((255, 255, 255), (0, 0, 0)) if pol else ((0, 0, 0), (255, 255, 255))
    c = Canvas(st_col, rng, "")
    S, G = rng.uniform(84, 96), rng.uniform(18, 24)
    P = S + G
    if cond == "none":                  # curved streets (Geier et al. 2008): no spots
        c.rect(-560, -420, 560, 420, sq_col)
        amp, lam = G * 0.6, P
        for k in range(-6, 7):
            y0 = k * P
            c.line([(x, y0 + amp * math.sin(2 * math.pi * x / lam)) for x in range(-560, 561, 6)], st_col, G)
            x0 = k * P
            c.line([(x0 + amp * math.sin(2 * math.pi * y / lam), y) for y in range(-420, 421, 6)], st_col, G)
    else:
        for i in range(-6, 6):
            for j in range(-6, 6):
                x, y = j * P + G / 2, i * P + G / 2
                c.rect(x, y, x + S, y + S, sq_col)
    return c.done(), yesno(True), {"square": round(S), "street": round(G)}


def scintillating_grid(cond, r, rng, d):
    c = Canvas((0, 0, 0), rng, "R90" if r % 2 else "")
    P, G, R = rng.uniform(86, 96), rng.uniform(14, 18), rng.uniform(12, 15)
    grey = (rng.randint(95, 115),) * 3
    rot = math.radians(45) if cond == "reverse" else 0
    def q(x, y):
        return (x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot))
    n = 8
    for k in range(-n, n + 1):
        c.line([q(k * P, -n * P), q(k * P, n * P)], grey, G)
        c.line([q(-n * P, k * P), q(n * P, k * P)], grey, G)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = (j * P, i * P) if cond != "none" else (j * P + P / 2, i * P)   # none: discs between crossings
            c.circle(*q(x, y), R, fill=(255, 255, 255))
    return c.done(), yesno(True), {}


# ---------------------------------------------------------------- C: colour

def munker_white(cond, r, rng, d):
    turn = "T" if r % 2 else ""
    c = Canvas((255, 255, 255), rng, turn)
    s = rng.uniform(16, 22)
    test = (235, 105, 30)
    blue, yellow, neutral = (40, 80, 255), (250, 235, 0), (120, 120, 120)
    ind = {"normal": (blue, yellow), "reverse": (yellow, blue), "none": (neutral, neutral)}[cond]
    rows = int(440 / s)
    top = -rows * s / 2
    for x0, col in ((-470, ind[0]), (30, ind[1])):
        for i in range(rows):
            c.rect(x0, top + i * s, x0 + 440, top + (i + 1) * s, test if i % 2 else col)
    return c.done(), yesno(False), {}


def color_contrast(cond, r, rng, d):
    turn = "T" if r % 2 else ""
    c = Canvas((255, 255, 255), rng, turn)
    red, green = (200, 40, 50), (40, 170, 70)
    grey = (rng.randint(120, 140),) * 3
    # the grey on the green ground looks reddish
    bgs = {"normal": (green, red), "reverse": (red, green), "none": ((190, 190, 190),) * 2}[cond]
    for x, b in ((-230, bgs[0]), (230, bgs[1])):
        c.rect(x - 220, -220, x + 220, 220, b)
        c.rect(x - 70, -70, x + 70, 70, grey)
    first, second = ("top", "bottom") if turn == "T" else ("left", "right")
    ill = {"normal": 0, "reverse": 1}.get(cond)
    return c.done(), three(first, second, "same", ill, 0), {}


def neon_spreading(cond, r, rng, d):
    dark_bg = cond == "reverse"
    bg, line_col = ((0, 0, 0), (255, 255, 255)) if dark_bg else ((255, 255, 255), (0, 0, 0))
    col = [(230, 30, 40), (30, 90, 240), (230, 30, 40), (30, 150, 60)][r]
    c = Canvas(bg, rng, "")
    P, w, R = rng.uniform(52, 62), lw(rng, 5), rng.uniform(150, 175)
    gap = 12 if cond == "none" else 0
    n = int(420 / P)

    def seg(p0, p1, horiz):
        # split a grid line at the circle: coloured inside, plain outside (gap in `none`)
        (x0, y0), (x1, y1) = p0, p1
        k = y0 if horiz else x0
        if abs(k) >= R:
            c.line([p0, p1], line_col, w)
            return
        e = math.sqrt(R * R - k * k)
        if horiz:
            c.line([(x0, k), (-e - gap, k)], line_col, w)
            c.line([(-e, k), (e, k)], col, w)
            c.line([(e + gap, k), (x1, k)], line_col, w)
        else:
            c.line([(k, y0), (k, -e - gap)], line_col, w)
            c.line([(k, -e), (k, e)], col, w)
            c.line([(k, e + gap), (k, y1)], line_col, w)

    for k in range(-n, n + 1):
        seg((-n * P - 40, k * P), (n * P + 40, k * P), True)
        seg((k * P, -n * P - 40), (k * P, n * P + 40), False)
    return c.done(), yesno(True), {}


# ---------------------------------------------------------------- K: completion

def kanizsa(cond, r, rng, d):
    pol = cond == "reverse"
    bg, fg = ((0, 0, 0), (255, 255, 255)) if pol else ((255, 255, 255), (0, 0, 0))
    c = Canvas(bg, rng, "R180" if r % 2 else "")
    R, Rc = rng.uniform(70, 82), rng.uniform(215, 235)
    for i in range(3):
        t = -math.pi / 2 + 2 * math.pi * i / 3
        x, y = Rc * math.cos(t), Rc * math.sin(t)
        to_c = math.atan2(-y, -x) + (math.pi if cond == "none" else 0)   # none: mouths turned outwards
        a0, a1 = math.degrees(to_c + math.pi / 6), math.degrees(to_c - math.pi / 6 + 2 * math.pi)
        px, py = c.p(x, y)
        rr = c.k(R)
        tr = c.turn == "R180"
        c.d.pieslice([px - rr, py - rr, px + rr, py + rr], a0 + (180 if tr else 0), a1 + (180 if tr else 0), fill=fg)
    return c.done(), yesno(True), {}


def ehrenstein(cond, r, rng, d):
    pol = cond == "reverse"
    bg, fg = ((0, 0, 0), (255, 255, 255)) if pol else ((255, 255, 255), (0, 0, 0))
    c = Canvas(bg, rng, "")
    n, r0, w = rng.choice([12, 16, 20]), rng.uniform(70, 90), lw(rng, 5)
    if cond == "none":
        r0 = 0
    rot = rng.uniform(0, math.pi)
    for i in range(n):
        t = rot + 2 * math.pi * i / n
        c.line([(r0 * math.cos(t), r0 * math.sin(t)), (340 * math.cos(t), 340 * math.sin(t))], fg, w)
    return c.done(), yesno(True), {"lines": n}


# ---------------------------------------------------------------- M: motion in still pictures

def rotating_snakes(cond, r, rng, d):
    pal = [((0, 0, 0), (30, 70, 210), (255, 255, 255), (235, 210, 30)),
           ((0, 0, 0), (90, 90, 90), (255, 255, 255), (175, 175, 175))][r % 2]
    c = Canvas((128, 128, 128), rng, "")
    widths = (0.18, 0.32, 0.18, 0.32)
    rot = rng.uniform(0, math.pi)
    for ring in range(6):
        r0, r1 = 50 + ring * 55, 50 + (ring + 1) * 55 - 4
        ncell = max(6, int(2 * math.pi * (r0 + r1) / 2 / 120))
        off = rot + (math.pi / ncell if ring % 2 else 0)
        for j in range(ncell):
            a = off + 2 * math.pi * j / ncell
            cw = 2 * math.pi / ncell
            order = [0, 1, 2, 3]
            if cond == "reverse" or (cond == "none" and j % 2):
                order = order[::-1]
            pos = a
            for k in order:
                c.sector(0, 0, r0, r1, pos, pos + cw * widths[k], pal[k])
                pos += cw * widths[k]
    # bands run black, dark, white, light with increasing image angle = clockwise on screen;
    # the motion goes from black to dark grey, i.e. clockwise
    ill = {"normal": "clockwise", "reverse": "counterclockwise"}.get(cond)
    return c.done(), {"options": ["clockwise", "counterclockwise", "still"], "correct": "still",
                      "illusion_answer": ill}, {}


def pinna_brelstaff(cond, r, rng, d):
    """After Pinna & Brelstaff (2000), read off the published figure: two rings of about 40 outlined parallelograms
    on light grey, with a fixation cross. Two sides of each element run along the ring, the other two are sheared
    (opposite shear in the two rings). In both rings the outer side is black and the inner side white; of the two
    sheared sides, the clockwise one is black in the outer ring and the counter-clockwise one in the inner ring.
    Moving towards the cross makes the rings counter-rotate. reverse = the two rings' patterns swapped;
    none = all sides black (no luminance polarity)."""
    c = Canvas((178, 178, 178), rng, "")
    BL, WH = (40, 40, 40), (255, 255, 255)
    n = rng.choice([38, 40, 42])
    phase = rng.uniform(0, 2 * math.pi)
    shear = math.radians(rng.uniform(25, 30))
    wl = 1.6
    rings = ((364, 27, 1), (286, 21, -1))               # radius, side, which sheared side is black (+1 = clockwise)
    if cond == "reverse":
        rings = ((364, 27, -1), (286, 21, 1))
    for R, side, cw_black in rings:
        h = side / 2
        sh = -math.tan(shear) * h * cw_black            # the inner side leans towards the black sheared side
        for i in range(n):
            a = phase + 2 * math.pi * i / n
            rx, ry = math.cos(a), math.sin(a)           # outwards
            tx, ty = -ry, rx                            # tangent; + = clockwise on screen (image y points down)
            def P(t, rr):                               # t along the ring, rr outwards from the ring
                return ((R + rr) * rx + t * tx, (R + rr) * ry + t * ty)
            # outer side shifted by +sh along the ring, inner side by -sh: a parallelogram (checked against the figure)
            o0, o1 = P(-h + sh, h), P(h + sh, h)
            i0, i1 = P(-h - sh, -h), P(h - sh, -h)
            if cond == "none":
                cols = (BL, BL, BL, BL)
            else:
                cols = (BL, WH, BL if cw_black > 0 else WH, WH if cw_black > 0 else BL)
            c.line([o0, o1], cols[0], wl)               # outer side
            c.line([i0, i1], cols[1], wl)               # inner side
            c.line([o1, i1], cols[2], wl)               # clockwise side
            c.line([o0, i0], cols[3], wl)               # counter-clockwise side
    c.line([(-12, 0), (12, 0)], BL, 2)
    c.line([(0, -12), (0, 12)], BL, 2)
    return c.done(), yesno(True), {"n": n, "shear": round(math.degrees(shear), 1)}


def ouchi(cond, r, rng, d):
    s = rng.uniform(0.9, 1.05)
    a, b = round(44 * s), round(11 * s)                 # long and short side of a check
    R = round(rng.uniform(150, 175))
    cx, cy = W // 2 + rng.randint(-15, 15), H // 2 + rng.randint(-12, 12)

    def tex(horiz, phase=0):
        im = Image.new("L", (W, H), 255)
        dd = ImageDraw.Draw(im)
        cw, ch = (a, b) if horiz else (b, a)
        for i in range(-1, H // ch + 2):
            for j in range(-1, W // cw + 2):
                if (i + j) % 2:
                    y0 = i * ch + (phase if horiz else 0)
                    x0 = j * cw + (0 if horiz else phase)
                    dd.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=0)
        return im

    surround_h = cond != "reverse"
    outer = tex(surround_h)
    inner = tex(not surround_h) if cond != "none" else tex(surround_h, phase=b)
    mask = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(mask).ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], fill=255)
    mask = mask.resize((W, H), Image.LANCZOS)
    out = Image.composite(inner, outer, mask).convert("RGB")
    return out, yesno(True), {"radius": R}


# ---------------------------------------------------------------- D: depth

def shepard_tables(cond, r, rng, d):
    fg, bg = SCHEMES[r]
    top_col = [(150, 95, 50), (190, 140, 80), (120, 80, 60), (160, 160, 160)][r]
    c = Canvas(bg, rng, "")
    w = lw(rng, 4)
    k1, k2 = pair(d)
    if cond == "none":
        for x, k in ((-250, k1), (250, k2)):
            c.poly([(x - 130 * k, -65), (x + 130 * k, -65), (x + 130 * k, 65), (x - 130 * k, 65)], top_col, fg, w)
        return c.done(), three("left", "right", "same", None, d), {}
    a, b = 300, 135
    th, ph = math.radians(72), math.radians(162)

    def table(cx, rot):
        u = (a * math.cos(th + rot), -a * math.sin(th + rot))
        v = (b * math.cos(ph + rot), -b * math.sin(ph + rot))
        p0 = (cx - (u[0] + v[0]) / 2, -40 - (u[1] + v[1]) / 2)
        pts = [p0, (p0[0] + u[0], p0[1] + u[1]), (p0[0] + u[0] + v[0], p0[1] + u[1] + v[1]), (p0[0] + v[0], p0[1] + v[1])]
        low = sorted(range(4), key=lambda i: -pts[i][1])[:3]
        thick = 16
        for i in range(4):
            j = (i + 1) % 4
            if i in low and j in low:
                q = [pts[i], pts[j], (pts[j][0], pts[j][1] + thick), (pts[i][0], pts[i][1] + thick)]
                c.poly(q, tuple(int(v * 0.7) for v in top_col), fg, w)
        for i in low:
            c.line([(pts[i][0], pts[i][1] + thick), (pts[i][0], pts[i][1] + thick + 120)], fg, w * 2.5)
        c.poly(pts, top_col, fg, w)

    # the top whose long side runs up the picture (into depth) looks long and narrow
    deep, flat = 0, -math.pi / 2
    left, right = (deep, flat) if cond == "normal" else (flat, deep)
    table(-270, left)
    table(270, right)
    return c.done(), three("left", "right", "same", 0 if cond == "normal" else 1, 0), {}


# ---------------------------------------------------------------- the list

ILLUSIONS = [
    # id, category, generator, resolution check in `none`, question (draft; prompts.py words it per rung)
    ("muller_lyer", "G", muller_lyer, True, "Which line looks longer?"),
    ("ponzo", "G", ponzo, True, "Which of the two short bars looks longer?"),
    ("ebbinghaus", "G", ebbinghaus, True, "Which central circle looks larger?"),
    ("delboeuf", "G", delboeuf, True, "Which filled circle looks larger?"),
    ("vertical_horizontal", "G", vertical_horizontal, True, "Which line looks longer, the vertical or the horizontal one?"),
    ("jastrow", "G", jastrow, True, "Which of the two shapes looks larger?"),
    ("poggendorff", "G", poggendorff, True, "Compared with where the left part of the slanted line would continue, the right part looks higher, lower, or aligned?"),
    ("zollner", "G", zollner, False, "Do the long lines look parallel to each other?"),
    ("cafe_wall", "G", cafe_wall, False, "Do the horizontal grey lines look parallel to each other?"),
    ("hering", "G", hering, False, "Do the two thick lines look straight?"),
    ("fraser_spiral", "G", fraser_spiral, False, "Does the pattern look like a spiral?"),
    ("simultaneous_contrast", "L", simultaneous_contrast, True, "Which small grey square looks lighter?"),
    ("checker_shadow", "L", checker_shadow, True, "Which square looks lighter, A or B?"),
    ("white_effect", "L", white_effect, True, "Which group of grey bars looks lighter?"),
    ("hermann_grid", "L", hermann_grid, False, "Do you see faint spots where the lines cross?"),
    ("scintillating_grid", "L", scintillating_grid, False, "Do you see dark dots inside the white discs?"),
    ("munker_white", "C", munker_white, False, "Do the orange stripes on the two sides look the same colour?"),
    ("color_contrast", "C", color_contrast, False, "Which small grey square looks more reddish?"),
    ("neon_spreading", "C", neon_spreading, False, "Do you see a faint coloured disc filling the centre, between the lines?"),
    ("kanizsa", "K", kanizsa, False, "Do you see a white triangle with clear edges lying on top of the black shapes?"),
    ("ehrenstein", "K", ehrenstein, False, "Do you see a disc in the centre where the lines would meet?"),
    ("rotating_snakes", "M", rotating_snakes, False, "Does the disc look like it is rotating? If so, which way?"),
    ("pinna_brelstaff", "M", pinna_brelstaff, False, "While looking at the central cross and moving your head towards and away from the picture, do the rings look like they rotate?"),
    ("ouchi", "M", ouchi, False, "Does the central disc look like it moves or floats separately from the background?"),
    ("shepard_tables", "D", shepard_tables, True, "Which tabletop looks longer and narrower?"),
]


def generate(out: Path = OUT) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (ill, cat, gen, check, question) in enumerate(ILLUSIONS, 1):
        for cond in CONDS:
            for r in range(RENDERS):
                d = 0.0
                if cond == "none" and check and r >= 2:
                    d = D if r == 2 else -D
                rng = random.Random(f"gorilla-{ill}-{cond}-{r}")   # seed string kept from the original run (same images)
                img, meta, params = gen(cond, r, rng, d)
                if cond == "none":
                    meta["illusion_answer"] = None      # nothing induces the illusion
                iid = f"{n:02d}_{ill}_{cond}_{r}"
                img.save(out / f"{iid}.png", optimize=True)
                rows.append({"id": iid, "n": n, "illusion": ill, "category": cat, "condition": cond, "render": r,
                             "diff": d, "question": question, "options": meta["options"], "correct": meta["correct"],
                             "illusion_answer": meta["illusion_answer"], "params": params})
    with open(out / "manifest.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["id", "n", "illusion", "category", "condition", "render", "diff", "question", "options", "correct",
                "illusion_answer", "params"]
        w.writerow(keys)
        for row in rows:
            w.writerow([" / ".join(row[k]) if k == "options" else json.dumps(row[k]) if k == "params" else row[k]
                        for k in keys])
    return rows


def contact_sheets(rows: list[dict], out: Path = OUT, per_sheet: int = 9, tw: int = 240) -> list[Path]:
    th = tw * H // W
    lab = 190
    f = ImageFont.truetype(FONT, 15)
    fs = ImageFont.truetype(FONT, 12)
    by_ill = {}
    for row in rows:
        by_ill.setdefault(row["n"], []).append(row)
    ns = sorted(by_ill)
    paths = []
    for k in range(0, len(ns), per_sheet):
        chunk = ns[k:k + per_sheet]
        sheet = Image.new("RGB", (lab + 12 * (tw + 6), 30 + len(chunk) * (th + 26)), (235, 235, 235))
        dd = ImageDraw.Draw(sheet)
        for ci, cond in enumerate(CONDS):
            dd.text((lab + ci * 4 * (tw + 6) + 2 * (tw + 6), 15), cond, font=f, fill=(0, 0, 0), anchor="mm")
        for i, n in enumerate(chunk):
            y = 30 + i * (th + 26)
            items = by_ill[n]
            dd.text((6, y + th // 2), f"{n}. {items[0]['illusion']}", font=f, fill=(0, 0, 0), anchor="lm")
            for j, row in enumerate(items):
                x = lab + j * (tw + 6)
                im = Image.open(out / f"{row['id']}.png").convert("RGB").resize((tw, th), Image.LANCZOS)
                sheet.paste(im, (x, y))
                cap = f"ill={row['illusion_answer'] or '-'} ok={row['correct']}"
                dd.text((x + 2, y + th + 3), cap, font=fs, fill=(150, 0, 0) if row["diff"] else (0, 0, 0))
        p = out / f"sheet_{k // per_sheet + 1}.png"
        sheet.save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    rows = generate(out)
    for p in contact_sheets(rows, out):
        print(p)
    print(f"{len(rows)} images in {out}")
