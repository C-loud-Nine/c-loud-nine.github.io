#!/usr/bin/env python3
r"""
make_gorent_video.py

A paper video for "GoRent: Open-source software for transparent rule-based
rental risk assessment and intervention support" (SoftwareX 35, 2026),
built straight from the published PDF. Figures are cropped from the source
at high DPI, so nothing is a screenshot and nothing is redrawn.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_gorent_video.py "gorent/gorent.pdf"

Default is a short loop: the app, the architecture, the scoring, the
conversation. --full adds the intervention policy, the pipeline, and title
and closing cards.

    --grid    one PNG per page with a labelled coordinate overlay, so the
              FIGURES boxes can be read off instead of guessed
    --themes  WCAG contrast table for every beat palette

Every claim on screen is quoted from the paper: Section 2 for the stack,
Fig. 2 for the six indicator weights, Fig. 3 for the band thresholds,
Table C for the licence and archive.
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        sys.exit("pip install pymupdf")

# ---------------------------------------------------------------- palette
# Each beat gets its own ground. A single background for the whole clip
# makes every beat read as the same slide with different art on it. The
# first is the site's own palette, so the poster frame stays on-brand.
THEMES = [
    dict(bg=(251, 246, 236), ink=(52, 48, 42), muted=(118, 112, 102),
         accent=(161, 61, 36), pine=(27, 58, 47), card=(255, 253, 250)),   # site paper
    dict(bg=(235, 240, 243), ink=(20, 30, 36), muted=(90, 104, 112),
         accent=(21, 104, 116), pine=(16, 64, 78), card=(255, 255, 255)),  # cool slate
    dict(bg=(245, 238, 231), ink=(36, 27, 22), muted=(118, 104, 94),
         accent=(160, 70, 38), pine=(70, 50, 36), card=(255, 255, 255)),   # clay
    dict(bg=(236, 241, 234), ink=(22, 32, 25), muted=(94, 108, 96),
         accent=(150, 80, 30), pine=(30, 70, 46), card=(255, 255, 255)),   # sage
    dict(bg=(243, 240, 246), ink=(28, 26, 36), muted=(110, 106, 120),
         accent=(124, 62, 128), pine=(48, 40, 78), card=(255, 255, 255)),  # lilac
]
THEME_NAMES = ["site paper", "cool slate", "clay", "sage", "lilac"]

BG, INK, MUTED = THEMES[0]["bg"], THEMES[0]["ink"], THEMES[0]["muted"]
ACCENT, PINE, CARD = THEMES[0]["accent"], THEMES[0]["pine"], THEMES[0]["card"]

W, H = 1920, 1080
SS = 2
FPS = 30
DPI = 400

FONTS = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
FONTS_B = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_fc = {}


def F(size, bold=False):
    k = (int(size), bold)
    if k not in _fc:
        for p in (FONTS_B if bold else FONTS):
            if os.path.exists(p):
                try:
                    _fc[k] = ImageFont.truetype(p, int(size))
                    break
                except OSError:
                    pass
        else:
            _fc[k] = ImageFont.load_default()
    return _fc[k]


# ------------------------------------------------- figure crops (page, box)
# Verified against the rendered pages; all seven land on their figure.
FIGURES = {
    "fig1_arch":     (1, 0.500, 0.232, 0.950, 0.533),
    "fig2_score":    (2, 0.490, 0.058, 0.950, 0.250),
    "fig3_policy":   (2, 0.495, 0.290, 0.950, 0.470),
    "fig4_pipeline": (2, 0.495, 0.485, 0.950, 0.700),
    "fig5_chat":     (3, 0.040, 0.055, 0.490, 0.295),
    "fig6_app":      (3, 0.495, 0.055, 0.950, 0.295),
    "table1":        (3, 0.505, 0.305, 0.935, 0.445),
}

# the six boxes of fig4, in flow order: the figure runs left-to-right along
# the top row, then right-to-left along the second
PIPELINE_STEPS = [
    (0.010, 0.040, 0.305, 0.300),   # user input
    (0.325, 0.040, 0.625, 0.300),   # preprocessing
    (0.645, 0.040, 0.990, 0.300),   # intent detection
    (0.645, 0.400, 0.990, 0.620),   # data retrieval
    (0.325, 0.400, 0.625, 0.620),   # risk calculation
    (0.010, 0.400, 0.305, 0.620),   # response generation
]

PIPELINE_CLOSE = "The same history and the same query give the same answer"

# ---------------------------------------------------------------- beats
BEATS = [
    dict(kind="title", seconds=4.0),
    dict(kind="figure", figure="fig1_arch", seconds=5.0,
         head="One system, not an analytics demo",
         sub="Flutter client, Go backend, MySQL persistence, Firebase notifications",
         chips=["three tiers", "REST API", "MIT licensed"]),
    dict(kind="figure", figure="fig2_score", seconds=5.5,
         head="The risk score is a weighted sum you can read",
         sub="Six payment-history indicators, fixed weights, bounded contributions",
         chips=["delay rate 30%", "average delay 25%", "no learned model"]),
    dict(kind="figure", figure="fig3_policy", seconds=4.5,
         head="Scores map to actions, not verdicts",
         sub="Four bands at 0.35, 0.65 and 0.85, each tied to an intervention",
         chips=["defaults, not prescriptions", "recalibrate per portfolio"]),
    dict(kind="pipeline", figure="fig4_pipeline", seconds=14.0,
         head="Asking in natural language, answered deterministically",
         sub="Regex intent detection and template responses: no training data, repeatable output",
         steps=["A manager types an ordinary question",
                "Identifiers are normalised, including Bangladeshi phone formats",
                "Intent is matched by pattern, not predicted by a model",
                "The relevant payment records are retrieved",
                "The same scoring function runs on them",
                "A templated answer with its contributing factors"]),
    dict(kind="figure", figure="fig5_chat", seconds=5.0,
         head="Tenant summaries, high-risk listings, monthly rollups",
         sub="Every answer carries the category, the reason and the suggested next action",
         chips=["same query, same answer", "anonymised examples"]),
    dict(kind="figure", figure="fig6_app", seconds=4.5,
         head="Running software, not a prototype screenshot",
         sub="Property management, notification feed and payment workflow on live backend state",
         chips=["Android and iOS", "Docker-assisted deployment"]),
    dict(kind="end", seconds=5.0,
         # the paper qualifies the 30 minutes: "Users with command-line and
         # Go/Flutter experience can deploy in 30min." Dropping the clause
         # turns a setup estimate into a claim about everyone.
         lines=["Deterministic scoring: the same history always yields the same score",
                "Every risk summary decomposes into its six contributing terms",
                "MIT licensed, archived on Zenodo, deployable in 30min with Go/Flutter experience"]),
]


# The default cut: four beats, ~10s, no title or end card, because the page
# around it already says whose paper this is. The running app leads, since
# a screenshot of working software is the one frame that reads at thumbnail
# size; the architecture and the scoring explain it afterwards.
SHORT = [
    dict(kind="figure", figure="fig6_app", seconds=2.4,
         head="Rental management that can explain itself",
         sub="",
         chips=["open source, MIT"]),
    dict(kind="figure", figure="fig1_arch", seconds=2.8,
         head="Three tiers, all released",
         sub="Flutter client, Go REST backend, MySQL persistence, Firebase notifications",
         chips=["Docker-assisted deploy"]),
    dict(kind="figure", figure="fig2_score", seconds=2.8,
         head="Risk is a weighted sum you can read",
         sub="Six payment indicators, fixed weights, no learned model",
         chips=["deterministic", "auditable"]),
    dict(kind="figure", figure="fig5_chat", seconds=2.4,
         head="Ask in plain language, get a traceable answer",
         sub="",
         chips=["category, reason, next action"]),
]


# ---------------------------------------------------------------- colour
def _lum(c):
    c = [x / 255 for x in c]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    l1, l2 = sorted((_lum(a), _lum(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def check_themes():
    """WCAG ratios for every foreground against its own ground.
    body text wants >= 4.5, large headings >= 3.0."""
    print("%-12s%7s%9s%7s%8s" % ("theme", "ink", "heading", "muted", "accent"))
    ok = True
    for n, th in zip(THEME_NAMES, THEMES):
        r = [contrast(th[k], th["bg"]) for k in ("ink", "pine", "muted", "accent")]
        flags = ["" if v >= t else " FAIL" for v, t in zip(r, (4.5, 3.0, 4.5, 3.0))]
        ok &= not any(flags)
        print("%-12s" % n + "".join("%7.1f%s" % (v, f) for v, f in zip(r, flags)))
    print("all pass" if ok else "SOME FAIL")
    return ok


def set_theme(i):
    global BG, INK, MUTED, ACCENT, PINE, CARD
    th = THEMES[i % len(THEMES)]
    BG, INK, MUTED = th["bg"], th["ink"], th["muted"]
    ACCENT, PINE, CARD = th["accent"], th["pine"], th["card"]


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def spring(t, overshoot=1.7):
    """Ease with a small overshoot. Chips that settle rather than stop dead
    read as physical."""
    t = max(0.0, min(1.0, t))
    return 1 + (overshoot + 1) * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def ease_io(t):
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def seg(t, a, b):
    return 0.0 if b <= a else max(0.0, min(1.0, (t - a) / (b - a)))


def fade(c, a):
    return tuple(c) + (int(max(0, min(255, a * 255))),)


def step_schedule(steps):
    """Screen time per pipeline step, proportional to its caption length.
    Uniform timing makes the longest caption the fastest to read."""
    w = [max(2.0, len(s.split())) for s in steps]
    tot = sum(w)
    out, acc = [], 0.0
    for x in w:
        out.append((acc / tot, (acc + x) / tot))
        acc += x
    return out


# ------------------------------------------------------------------- io
def crop_figure(doc, spec, dpi=DPI):
    page, x0, y0, x1, y1 = spec
    pix = doc[page].get_pixmap(dpi=dpi, alpha=False)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return im.crop((int(x0 * im.width), int(y0 * im.height),
                    int(x1 * im.width), int(y1 * im.height)))


def trim_white(im, thresh=248, pad=10):
    """Drop the white margin the crop inevitably includes, so the figure
    fills its slot rather than floating in a box of nothing."""
    a = np.asarray(im.convert("L"))
    mask = a < thresh
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))


def grid_pages(doc, folder):
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=120, alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        d = ImageDraw.Draw(im)
        for k in range(1, 20):
            x = im.width * k / 20
            d.line([(x, 0), (x, im.height)], fill=(255, 150, 150))
            d.text((x + 2, 2), "%.2f" % (k / 20), fill=(200, 0, 0), font=F(11, True))
            y = im.height * k / 20
            d.line([(0, y), (im.width, y)], fill=(150, 150, 255))
            d.text((2, y + 2), "%.2f" % (k / 20), fill=(0, 0, 200), font=F(11, True))
        im.save(os.path.join(folder, "page%d_grid.png" % i))
    print("coordinate overlays written next to the PDF")


_shadow_cache = {}


def shadow(size, radius=16, blur=26, alpha=58):
    """Memoised: card geometry never changes, and blurring it once per frame
    was the largest single cost in the render."""
    key = (size, radius, blur, alpha)
    if key not in _shadow_cache:
        w, h = size
        lay = Image.new("RGBA", (w + 4 * blur, h + 4 * blur), (0, 0, 0, 0))
        ImageDraw.Draw(lay).rounded_rectangle(
            [2 * blur, 2 * blur + blur // 2, 2 * blur + w, 2 * blur + blur // 2 + h],
            radius=radius, fill=(0, 0, 0, alpha))
        _shadow_cache[key] = (lay.filter(ImageFilter.GaussianBlur(blur / 2)),
                              2 * blur, 2 * blur)
    return _shadow_cache[key]


def round_corners(im, r=12):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, im.width, im.height], radius=r, fill=255)
    out = im.convert("RGBA")
    out.putalpha(m)
    return out


def chip(d, xy, text, size, fg, bg, a=1.0):
    f = F(size, True)
    pad = size * 0.55
    w = d.textlength(text, font=f) + 2 * pad
    h = size + 1.5 * pad
    x, y = xy
    d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=fade(bg, a))
    d.text((x + pad, y + pad * 0.72), text, font=f, fill=fade(fg, a))
    return w


def header(d, S, args, a=1.0, small=False):
    size = 44 if small else 56
    d.text((86 * S, 40 * S), args.title, font=F(size * S, True), fill=fade(PINE, a))
    d.text((86 * S, (40 + size + 10) * S), args.venue, font=F(28 * S), fill=fade(MUTED, a))


_stage_cache = {}


def stage_figure(fig, top, bottom, pad=120):
    """Figure scaled to fit the band [top, bottom]. Cached: this ran a
    LANCZOS resize of a 400-DPI crop on every single frame."""
    key = (id(fig), fig.size, top, bottom, pad)
    if key not in _stage_cache:
        avail_w, avail_h = W - 2 * pad, bottom - top
        s = min(avail_w / fig.width, avail_h / fig.height)
        fw, fh = int(fig.width * s), int(fig.height * s)
        _stage_cache[key] = (fig.resize((fw, fh), Image.LANCZOS),
                             (W - fw) // 2, top + (avail_h - fh) // 2, fw, fh)
    return _stage_cache[key]


# ---------------------------------------------------------------- frames
def frame_title(args, t):
    canvas = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    a1 = ease_out(seg(t, 0.05, 0.45))
    dy = (1 - a1) * 26
    f = F(58 * S, True)
    for i, line in enumerate(args.title_lines):
        tw = d.textlength(line, font=f) / S
        d.text((((W - tw) / 2) * S, ((300 + i * 78) - dy) * S), line, font=f,
               fill=fade(PINE, a1))
    a2 = ease_out(seg(t, 0.25, 0.6))
    f2 = F(26 * S)
    for txt, y in ((args.authors, 486), (args.affil, 526)):
        tw = d.textlength(txt, font=f2) / S
        d.text((((W - tw) / 2) * S, y * S), txt, font=f2, fill=fade(MUTED, a2))
    a3 = ease_out(seg(t, 0.45, 0.8))
    f3 = F(26 * S, True)
    tw = d.textlength(args.venue, font=f3) / S
    cw = tw + 56
    d.rounded_rectangle([((W - cw) / 2) * S, 606 * S, ((W + cw) / 2) * S, 664 * S],
                        radius=29 * S, fill=fade(PINE, a3))
    d.text((((W - tw) / 2) * S, 619 * S), args.venue, font=f3,
           fill=fade((255, 255, 255), a3))
    rw = ease_out(seg(t, 0.15, 0.75)) * 160
    d.rectangle([((W - rw) / 2) * S, 248 * S, ((W + rw) / 2) * S, 251 * S],
                fill=fade(ACCENT, a1))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_figure(beat, fig, args, t):
    canvas = Image.new("RGB", (W, H), BG)

    # continuous slow zoom across the beat, plus a little drift. A still
    # image held for three seconds reads as a slide; 4.5% of travel is
    # enough to feel alive without anyone noticing the movement.
    # The zoom grows the figure INTO its slot rather than past it. Scaling
    # up and centre-cropping loses ~2% off every edge, which clipped the
    # sub-captions ("(a) Property management") off the bottom of Figs 5-6.
    ZOOM = 1.045
    scale = (1.0 + (ZOOM - 1.0) * ease_io(t)) / ZOOM
    drift = int(10 * ease_io(t))
    base, fx, fy, fw, fh = stage_figure(fig, 318, H - 232)
    iw, ih = max(1, int(fw * scale)), max(1, int(fh * scale))
    zoomed = base.resize((iw, ih), Image.LANCZOS)

    a = ease_out(seg(t, 0.04, 0.40))
    if a > 0.02:
        pad = 18
        plate = Image.new("RGB", (fw + 2 * pad, fh + 2 * pad), CARD)
        plate.paste(zoomed, (pad + (fw - iw) // 2, pad + (fh - ih) // 2))
        card = round_corners(plate, 16)
        sh, ox, oy = shadow(plate.size, alpha=62)
        dy = int((1 - a) * 34) - drift          # travels up, then drifts on
        if a > 0.25:
            sl = sh.copy()
            sl.putalpha(sl.getchannel("A").point(lambda v: int(v * a)))
            canvas.paste(sl, (fx - pad - ox, fy + dy - pad - oy), sl)
        px, py = fx - pad, fy + dy - pad
        if a < 1.0:
            base_c = canvas.crop((px, py, px + card.width, py + card.height)).convert("RGBA")
            card = Image.blend(base_c, card, a)
        canvas.paste(card, (px, py), card)

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    header(d, S, args, 1.0, small=True)

    ah = ease_out(seg(t, 0.0, 0.26))
    hx = 86 - (1 - ah) * 34
    d.text((hx * S, 162 * S), beat["head"], font=F(70 * S, True), fill=fade(PINE, ah))
    if beat.get("sub"):
        asub = ease_out(seg(t, 0.10, 0.36))
        sx = 86 - (1 - asub) * 22
        d.text((sx * S, 248 * S), beat["sub"], font=F(34 * S), fill=fade(MUTED, asub))

    rw = ease_out(seg(t, 0.06, 0.42)) * 120
    d.rectangle([86 * S, 144 * S, (86 + rw) * S, 149 * S], fill=fade(ACCENT, 1))

    cx = 86
    for i, c in enumerate(beat.get("chips", [])):
        ca = seg(t, 0.42 + i * 0.09, 0.66 + i * 0.09)
        if ca <= 0.02:
            continue
        sp = spring(ca)
        cx += chip(d, (cx * S, (H - 132 + (1 - sp) * 26) * S), c, 34 * S,
                   PINE, CARD, min(1.0, ca * 2)) / S + 16

    d.rectangle([0, (H - 5) * S, int(W * t) * S, H * S], fill=fade(ACCENT, 0.55))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_pipeline(beat, fig, args, t):
    canvas = Image.new("RGB", (W, H), BG)
    im, fx, fy, fw, fh = stage_figure(fig, 296, H - 196, pad=70)
    n = len(PIPELINE_STEPS)
    assert len(beat["steps"]) == n, "one caption per highlighted box"
    intro, outro = 0.08, 0.10
    body = 1.0 - intro - outro
    sched = step_schedule(beat["steps"])
    u = (t - intro) / body if body > 0 else 0.0
    k, local = 0, 0.0
    if t >= intro:
        for i, (a0, a1) in enumerate(sched):
            if u < a1 or i == n - 1:
                k, local = i, min(1.0, max(0.0, (u - a0) / (a1 - a0)))
                break
    final = t > 1 - outro

    reveal = Image.new("L", (fw, fh), 0)
    rd = ImageDraw.Draw(reveal)
    if final:
        reveal = Image.new("L", (fw, fh), 255)
    elif t >= intro:
        for i in range(k + 1):
            av = 255 if i < k else int(255 * ease_out(seg(local, 0, 0.45)))
            x0, y0, x1, y1 = PIPELINE_STEPS[i]
            rd.rounded_rectangle([x0 * fw, y0 * fh, x1 * fw, y1 * fh], radius=12, fill=av)
        reveal = reveal.filter(ImageFilter.GaussianBlur(22))
    dim = Image.new("RGB", (fw, fh), BG)
    shown = Image.composite(im, Image.blend(im, dim, 0.88), reveal)
    ia = ease_out(seg(t, 0, intro))
    if ia < 1:
        shown = Image.blend(Image.new("RGB", (fw, fh), BG), shown, ia)
    canvas.paste(shown, (fx, fy))

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    header(d, S, args, 1.0, small=True)
    d.text((86 * S, 162 * S), beat["head"], font=F(70 * S, True), fill=fade(PINE, 1))
    d.text((86 * S, 248 * S), beat.get("sub", ""), font=F(34 * S), fill=fade(MUTED, 1))

    if not final and t >= intro:
        oa = ease_out(seg(local, 0, 0.4)) * (1 - ease_out(seg(local, 0.9, 1.0)))
        x0, y0, x1, y1 = PIPELINE_STEPS[k]
        d.rounded_rectangle([(fx + x0 * fw) * S, (fy + y0 * fh) * S,
                             (fx + x1 * fw) * S, (fy + y1 * fh) * S],
                            radius=10 * S, outline=fade(ACCENT, oa), width=5 * S)
    cap = PIPELINE_CLOSE if final else beat["steps"][k]
    ca = 1.0 if final else ease_out(seg(local, 0.04, 0.26))
    f = F(42 * S, True)
    tw = d.textlength(cap, font=f) / S
    bx, by = (W - tw) / 2 - 36, H - 168
    d.rounded_rectangle([bx * S, by * S, (bx + tw + 72) * S, (by + 84) * S],
                        radius=42 * S, fill=fade(CARD, ca))
    d.text(((bx + 36) * S, (by + 20) * S), cap, font=f, fill=fade(PINE, ca))
    for i in range(n):
        r = 6 * S
        x = (W / 2 - (n - 1) * 14 + i * 28) * S
        y = (H - 52) * S
        on = i <= k and t >= intro
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=fade(ACCENT if (i == k and not final) else (PINE if on else MUTED),
                            1.0 if on else 0.3))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_end(beat, args, t):
    canvas = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    a = ease_out(seg(t, 0.02, 0.3))
    d.text((150 * S, 210 * S), "What it gives you", font=F(44 * S, True), fill=fade(PINE, a))
    d.rectangle([150 * S, 288 * S, (150 + a * 150) * S, 291 * S], fill=fade(ACCENT, a))
    for i, line in enumerate(beat["lines"]):
        la = ease_out(seg(t, 0.18 + i * 0.13, 0.5 + i * 0.13))
        if la <= 0.02:
            continue
        y = 370 + i * 92
        d.ellipse([152 * S, (y + 12) * S, 170 * S, (y + 30) * S], fill=fade(ACCENT, la))
        d.text((196 * S, y * S), line, font=F(28 * S), fill=fade(INK, la))
    fa = ease_out(seg(t, 0.55, 0.85))
    d.text((150 * S, 760 * S), args.doi, font=F(26 * S, True), fill=fade(PINE, fa))
    d.text((150 * S, 806 * S), args.authors + "  .  " + args.affil,
           font=F(22 * S), fill=fade(MUTED, fa))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--title", default="GoRent")
    ap.add_argument("--title-lines", nargs="+",
                    default=["GoRent",
                             "Transparent rule-based rental risk assessment"])
    ap.add_argument("--authors", default="Sumaiya Rahim Suma, Abdullah Al Shafi")
    ap.add_argument("--affil", default="Khulna University of Engineering & Technology")
    ap.add_argument("--venue", default="SoftwareX 35 (2026) 102795")
    ap.add_argument("--doi", default="doi.org/10.1016/j.softx.2026.102795")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every beat's duration")
    ap.add_argument("--full", action="store_true",
                    help="the long cut: 8 beats with title and closing cards")
    ap.add_argument("--themes", action="store_true",
                    help="print the contrast table for every theme and exit")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default="gorent")
    args = ap.parse_args()

    if args.themes:
        sys.exit(0 if check_themes() else 1)
    if not args.pdf:
        sys.exit("give the paper PDF path")

    doc = fitz.open(args.pdf)
    folder = os.path.dirname(os.path.abspath(args.pdf))
    if args.grid:
        grid_pages(doc, folder)
        return

    beats = BEATS if args.full else SHORT
    figs = {}
    for beat in beats:
        name = beat.get("figure")
        if not name or name in figs:
            continue
        figs[name] = trim_white(crop_figure(doc, FIGURES[name]))
        print("  %-16s %dx%d" % (name, figs[name].width, figs[name].height))

    xf = max(4, int(0.3 * args.fps))
    plan = [(b, max(xf * 2, int(b["seconds"] * args.scale * args.fps))) for b in beats]
    xs = [max(3, min(xf, n // 4)) for _, n in plan]
    L = len(plan)
    count = sum(n for _, n in plan) - sum(xs)

    def render(b, i, n, theme):
        set_theme(theme)
        t = i / n
        if b["kind"] == "title":
            return frame_title(args, t)
        if b["kind"] == "end":
            return frame_end(b, args, t)
        if b["kind"] == "pipeline":
            return frame_pipeline(b, figs[b["figure"]], args, t)
        return frame_figure(b, figs[b["figure"]], args, t)

    def push(a_img, b_img, u):
        """Slide the outgoing frame left while the incoming one follows it
        in. Reads as a cut with intent rather than a dissolve."""
        out = Image.new("RGB", (W, H), BG)
        out.paste(a_img, (-int(W * 0.25 * u), 0))
        out.paste(b_img, (W - int(W * u), 0))
        return Image.blend(out, b_img, u ** 2.2)

    def stream():
        for j, (b, n) in enumerate(plan):
            nb, nn = plan[(j + 1) % L]
            # body starts at the PREVIOUS beat's transition length: those
            # opening frames already played inside the incoming push, and
            # replaying them stutters the beat
            for i in range(xs[(j - 1) % L], n - xs[j]):
                yield render(b, i, n, j)
            for k in range(xs[j]):
                u = ease_io((k + 1) / (xs[j] + 1))
                yield push(render(b, n - xs[j] + k, n, j),
                           render(nb, k, nn, (j + 1) % L), u)

    print("%d beats, %d frames, %dx%d, %.1fs" % (L, count, W, H, count / args.fps))

    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")
    stem = os.path.join(folder, args.out)
    # H.264 only: the one codec every browser plays, and +faststart lets it
    # start before it has finished downloading.
    mp4 = imageio.get_writer(stem + ".mp4", fps=args.fps, codec="libx264",
                             macro_block_size=1, pixelformat="yuv420p", quality=None,
                             output_params=["-crf", str(args.crf), "-preset", "slow",
                                            "-profile:v", "high",
                                            "-movflags", "+faststart", "-an"])
    for i, f in enumerate(stream()):
        mp4.append_data(np.asarray(f))
        if i % 60 == 0:
            print("    frame %d/%d" % (i, count), end="\r", flush=True)
    mp4.close()

    # poster rendered directly rather than sampled from the stream: a fixed
    # fraction of the clip lands wherever the beat boundaries happen to be,
    # and a transition frame posters as a double exposure
    pb, pn = plan[0]
    render(pb, int(pn * 0.86), pn, 0).save(stem + "_poster.jpg", quality=88,
                                           optimize=True, progressive=True)
    print(" " * 34, end="\r")
    for f in (stem + ".mp4", stem + "_poster.jpg"):
        if os.path.exists(f):
            print("  %-26s %9.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))


if __name__ == "__main__":
    main()
