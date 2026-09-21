#!/usr/bin/env python3
r"""
make_bangla_video.py

A paper video for "Semi-Supervised Context-Aware Emotion Classification
for Noisy Bangla Text Using Advanced Learning Strategies"
(WIECON-ECE 2025), built straight from the paper PDF. Figures are
cropped from the source at high DPI; the tables are typeset by the
script from values checked against the paper's own text layer.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_bangla_video.py "papers/bangla.pdf"

The framing throughout is label efficiency, not leaderboard position:
the proposed models use 15% labelled data against baselines that use
100%, and one of those baselines (Bhattacharjee et al., 0.682 macro F1)
is still ahead. Saying otherwise would misreport Table IV.

    --grid     one PNG per page with a labelled coordinate overlay
    --themes   WCAG contrast table for the chosen palette
    --verify   check every TABLES value against the PDF text
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
# One pastel family per paper. Within a family the beats shift shade
# slightly; across families they differ in hue.
PALETTES = {
    "sage":  [(237, 242, 235), (231, 238, 231), (241, 245, 238)],
    "paper": [(248, 245, 239), (245, 240, 232), (250, 247, 241)],
    "slate": [(236, 241, 244), (231, 237, 241), (240, 244, 247)],
    "clay":  [(246, 239, 232), (243, 235, 227), (248, 242, 236)],
    "lilac": [(243, 240, 246), (239, 236, 244), (246, 243, 249)],
}
INKS = {
    "sage":  dict(ink=(22, 32, 25), muted=(88, 102, 90), accent=(140, 74, 28), pine=(30, 70, 46)),
    "paper": dict(ink=(28, 30, 32), muted=(103, 99, 93), accent=(161, 61, 36), pine=(28, 58, 46)),
    "slate": dict(ink=(20, 30, 36), muted=(88, 102, 110), accent=(21, 104, 116), pine=(16, 64, 78)),
    "clay":  dict(ink=(36, 27, 22), muted=(110, 97, 88), accent=(150, 66, 36), pine=(70, 50, 36)),
    "lilac": dict(ink=(28, 26, 36), muted=(104, 100, 114), accent=(116, 58, 120), pine=(48, 40, 78)),
}
THEMES = []


def build_palette(name):
    global THEMES
    base = INKS.get(name, INKS["paper"])
    THEMES = [dict(bg=bg, card=bg, **base) for bg in PALETTES.get(name, PALETTES["paper"])]


BG, INK, MUTED = (247, 244, 238), (26, 30, 34), (124, 119, 111)
ACCENT, PINE, CARD = (200, 74, 38), (27, 58, 47), (255, 255, 255)

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


# ------------------------------------------------- tables, as data
# Typeset by the script rather than cropped: a screenshot of a table cannot
# animate, cannot be re-aligned, and carries the paper's body font into a
# 1920x1080 frame at the wrong size. Every value below was checked against
# the PDF text layer -- see --verify.
TABLES = {
    "sota": dict(
        title="Against published Bangla emotion work",
        cols=[("Method", "l", False), ("Macro F1", "r", True),
              ("Weighted F1", "r", True), ("Labelled", "r", False)],
        rows=[["Das et al. 2021", 0.621, 0.635, "100%"],
              ["Islam et al. 2022", 0.651, 0.663, "100%"],
              ["Sourav et al. 2022", 0.658, 0.672, "100%"],
              ["Bhattacharjee et al. 2022", 0.682, 0.695, "100%"],
              ["Ours: Teacher-Student + EPLF", 0.657, 0.662, "15%"],
              ["Ours: GAN-BERT", 0.662, 0.668, "15%"]],
        ours=5, unit="", dp=3,
        note="Within 0.02 macro F1 of the fully supervised best, on 15% of the labels"),
    "ablation": dict(
        title="Ablation on a 15% labelled seed (BanglaBERT-Large)",
        cols=[("SSL variant", "l", False), ("Macro F1", "r", True),
              ("Weighted F1", "r", True), ("Accuracy", "r", True)],
        rows=[["Teacher (seed only)", 0.613, 0.619, 0.623],
              ["Student (SSL)", 0.641, 0.646, 0.650],
              ["Student (calibrated)", 0.643, 0.648, 0.651],
              ["Student (FixMatch)", 0.645, 0.651, 0.653],
              ["Student (combined)", 0.657, 0.662, 0.665],
              ["GAN-BERT", 0.662, 0.668, 0.669]],
        ours=5, unit="", dp=3,
        note="Pseudo-labelling lifts the teacher by 0.044; GAN-BERT adds a further 0.005"),
    "seed": dict(
        title="What the labels cost",
        cols=[("Encoder", "l", False), ("Full data", "r", True), ("15% seed", "r", True)],
        rows=[["XLM-RoBERTa-Base", 0.677, 0.568],
              ["BanglaBERT-Base", 0.686, 0.550],
              ["BanglaBERT-Large", 0.702, 0.581]],
        ours=2, unit="", dp=3,
        note="Test F1. Every encoder gives up 0.11 to 0.14 when the labels are cut to 15%"),
}

# ------------------------------------------------- figure crops (page, box)
# Verified against the rendered pages; all three land on their figure.
FIGURES = {
    "fig1_wordcloud": (1, 0.505, 0.160, 0.935, 0.420),
    "fig2_preproc":   (2, 0.085, 0.078, 0.500, 0.162),
    "fig3_teacher":   (2, 0.100, 0.497, 0.480, 0.608),
    "fig4_pseudo":    (2, 0.495, 0.185, 0.880, 0.315),
    "fig5_ganbert":   (3, 0.095, 0.085, 0.495, 0.255),
    "fig6_calib":     (4, 0.498, 0.092, 0.940, 0.322),
    "fig7_confusion": (4, 0.505, 0.395, 0.935, 0.625),
    "fig8_curves":    (4, 0.498, 0.692, 0.940, 0.862),
}

# The word cloud is a raster of coloured type; repainting its ground would
# eat the pale words. Everything else here is line art or a plot.
NO_TINT = {"fig1_wordcloud"}

# Stages of fig3, read off a grid overlay of the TRIMMED figure.
#
# Each stage is a LIST of boxes. The first version put the unlabelled pool
# at y 0.02-0.93 and the teacher at y 0.38-0.93 in the same x band, so the
# teacher box sat 100% inside the pool box and the two steps rendered as an
# identical frame.
PIPELINE_STEPS = [
    [(0.005, 0.34, 0.145, 0.80)],                        # labelled seed data
    [(0.190, 0.02, 0.355, 0.37)],                        # unlabelled pool
    [(0.200, 0.39, 0.340, 0.94)],                        # teacher model
    [(0.398, 0.39, 0.572, 0.80)],                        # pseudo-label generation
    [(0.615, 0.50, 0.675, 0.72)],                        # concatenate seed + accepted
    [(0.720, 0.39, 0.885, 0.94), (0.900, 0.43, 1.000, 0.84)],   # student + output
]

PIPELINE_CLOSE = "The student ends up ahead of the teacher that taught it"

# ---------------------------------------------------------------- beats
BEATS = [
    dict(kind="title", seconds=4.0),
    dict(kind="figure", figure="fig1_wordcloud", seconds=4.5,
         head="22,698 noisy Bangla comments",
         sub="Code-mixing, transliteration, emoji and spelling variation across 12 domains",
         # the corpus is annotated with six emotions; the paper's experiments
         # use joy, love and sadness. It never says these were picked as the
         # "top three", so the chip does not claim a selection rule.
         chips=["EmoNoBa corpus", "six emotions annotated", "joy, love, sadness used"]),
    dict(kind="table", table="seed", seconds=6.0,
         head="Labels are the bottleneck",
         sub="Test F1 for the same encoders, trained on everything and on a 15% seed"),
    dict(kind="pipeline", figure="fig3_teacher", seconds=15.0,
         head="Teach a student from labels the teacher invents",
         sub="Enhanced Pseudo-Labeling Framework: calibration-aware thresholds, then balancing",
         steps=["A 15% labelled seed, and a large unlabelled pool",
                "The unlabelled pool is far larger than the seed",
                "The teacher trains on the seed with focal loss",
                "It pseudo-labels the pool: FixMatch, UDA, calibrated thresholds",
                "Seed and accepted labels are concatenated",
                "The student trains on the union and ends up ahead"]),
    dict(kind="figure", figure="fig4_pseudo", seconds=5.5,
         head="Four labellers, then a filter",
         sub="FixMatch, UDA, FlexMatch and calibration-aware branches, scored and selected",
         chips=["agreement raises reliability", "topic-emotion balancing"]),
    dict(kind="figure", figure="fig6_calib", seconds=5.5,
         head="Why the threshold is per class",
         sub="Calibration error differs sharply by emotion, so one global cut-off would misfire",
         chips=["Joy ECE 0.182", "Love 0.342", "Sadness 0.418"]),
    dict(kind="figure", figure="fig5_ganbert", seconds=5.5,
         head="Or skip pseudo-labels entirely",
         sub="GAN-BERT regularises the embedding space adversarially instead",
         chips=["generator makes synthetic embeddings", "discriminator separates real from fake"]),
    dict(kind="table", table="ablation", seconds=7.5,
         head="What each piece is worth",
         sub="All variants on the same 15% seed; GAN-BERT is a separate track"),
    dict(kind="table", table="sota", seconds=8.0,
         head="Against published work",
         sub="Everyone else trains on 100% of the labels"),
    dict(kind="figure", figure="fig7_confusion", seconds=5.0,
         head="Where it still goes wrong",
         sub="Love is the weakest class at 47% recall; Joy leaks into Sadness more than into Love",
         chips=["Sadness cleanest at 78%", "three classes, seed-controlled"]),
    dict(kind="end", seconds=5.0,
         lines=["15% of the labels, within 0.02 macro F1 of fully supervised work",
                "Calibration-aware thresholds, not a single global confidence cut",
                "Adversarial regularisation matches the best pseudo-label ensemble"]),
]


# The default cut: three beats, no title or end card, because the page
# around it already says whose paper this is. The method is the
# contribution, the table is the point, GAN-BERT is the alternative route.
SHORT = [
    dict(kind="table", table="sota", seconds=6.5,
         head="15% of the labels",
         sub="Everyone else trains on 100% of them"),
    dict(kind="figure", figure="fig7_confusion", seconds=3.6,
         head="Love is the class that still goes wrong",
         sub="",
         chips=["Sadness cleanest at 78%"]),
    dict(kind="pipeline", figure="fig3_teacher", seconds=8.0,
         head="A student taught by labels the teacher invents",
         sub="Calibration-aware pseudo-labelling on a 15% seed",
         steps=["A 15% labelled seed", "A large unlabelled pool",
                "Teacher trains on the seed", "Pseudo-labels, filtered by calibration",
                "Seed and accepted labels joined", "Student trains on the union"]),
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
    print("%-12s%7s%9s%7s%8s" % ("shade", "ink", "heading", "muted", "accent"))
    ok = True
    for i, th in enumerate(THEMES):
        r = [contrast(th[k], th["bg"]) for k in ("ink", "pine", "muted", "accent")]
        flags = ["" if v >= t else " FAIL" for v, t in zip(r, (4.5, 3.0, 4.5, 3.0))]
        ok &= not any(flags)
        print("%-12s" % ("shade %d" % (i + 1)) + "".join("%7.1f%s" % (v, f)
                                                         for v, f in zip(r, flags)))
    print("all pass" if ok else "SOME FAIL")
    return ok


def set_theme(i):
    global BG, INK, MUTED, ACCENT, PINE, CARD
    th = THEMES[i % len(THEMES)]
    BG, INK, MUTED = th["bg"], th["ink"], th["muted"]
    ACCENT, PINE, CARD = th["accent"], th["pine"], th["card"]


def mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


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


def verify_tables(doc):
    """Every printed value must appear in the paper's text layer.

    Blind spot worth knowing: this cannot see numbers rasterised INSIDE a
    figure. The ECE values on the calibration beat are real -- they are plot
    legends in Fig. 6 -- but they will never appear here, so chips are not
    checked, only TABLES.
    """
    import re
    flat = re.sub(r"\s+", " ", " ".join(p.get_text() for p in doc))
    bad = []
    for key, spec in TABLES.items():
        for row in spec["rows"]:
            for v in row[1:]:
                if isinstance(v, str):   # a label like "100%", not a score
                    continue
                dp = spec.get("dp", 2)
                if ("%.*f" % (dp, v)) not in flat:
                    bad.append("%s / %s / %.2f" % (key, row[0], v))
    if bad:
        print("VALUES NOT FOUND IN THE PDF:")
        for b in bad:
            print("   ", b)
        return False
    n = sum(len(s["rows"]) * (len(s["cols"]) - 1) for s in TABLES.values())
    print("  %d table values all present in the paper text" % n)
    return True


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


_TINT_CACHE = {}


def tint_to_bg(im, bg, cut=243):
    """Repaint a figure's white paper ground in the current tint. Left white,
    every crop announces itself as a screenshot pasted onto the slide. Only
    near-neutral BRIGHT pixels move, so coloured blocks and black text keep
    their values -- but a white mask is neutral and bright too, which is why
    photographic figures are excluded via NO_TINT rather than relying on the
    keying alone."""
    key = (id(im), bg, cut)
    if key not in _TINT_CACHE:
        a = np.asarray(im.convert("RGB")).astype(np.float32)
        mx, mn = a.max(2), a.min(2)
        # 14 rather than 30: a pale blue heat-map cell is low-saturation
        # too, and repainting it changes what the figure says
        w = (np.clip((mx - cut) / (255.0 - cut), 0, 1) * ((mx - mn) < 14))
        w = w.astype(np.float32)[..., None]
        _TINT_CACHE[key] = Image.fromarray(
            (a * (1 - w) + np.asarray(bg, np.float32) * w).astype(np.uint8))
    return _TINT_CACHE[key]


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
    f = F(54 * S, True)
    for i, line in enumerate(args.title_lines):
        tw = d.textlength(line, font=f) / S
        d.text((((W - tw) / 2) * S, ((300 + i * 76) - dy) * S), line, font=f,
               fill=fade(PINE, a1))
    a2 = ease_out(seg(t, 0.25, 0.6))
    f2 = F(24 * S)
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
    if not beat.get("_no_tint"):
        fig = tint_to_bg(fig, CARD)

    # The zoom grows the figure INTO its slot rather than past it. Scaling
    # up and centre-cropping loses ~2% off every edge, which clips the
    # column headings off figures that carry them.
    ZOOM = 1.045
    scale = (1.0 + (ZOOM - 1.0) * ease_io(t)) / ZOOM
    drift = int(10 * ease_io(t))
    base, fx, fy, fw, fh = stage_figure(fig, 318, H - 232)
    iw, ih = max(1, int(fw * scale)), max(1, int(fh * scale))
    zoomed = base.resize((iw, ih), Image.LANCZOS)

    a = ease_out(seg(t, 0.04, 0.40))
    if a > 0.02:
        pad = 24
        plate = Image.new("RGB", (fw + 2 * pad, fh + 2 * pad), CARD)
        plate.paste(zoomed, (pad + (fw - iw) // 2, pad + (fh - ih) // 2))
        card = round_corners(plate, 16)
        sh, ox, oy = shadow(plate.size, alpha=62)
        dy = int((1 - a) * 34) - drift
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
    d.text(((86 - (1 - ah) * 34) * S, 162 * S), beat["head"], font=F(70 * S, True),
           fill=fade(PINE, ah))
    if beat.get("sub"):
        asub = ease_out(seg(t, 0.10, 0.36))
        d.text(((86 - (1 - asub) * 22) * S, 248 * S), beat["sub"], font=F(34 * S),
               fill=fade(MUTED, asub))
    d.rectangle([86 * S, 144 * S, (86 + ease_out(seg(t, 0.06, 0.42)) * 120) * S, 149 * S],
                fill=fade(ACCENT, 1))

    cx = 86
    for i, c in enumerate(beat.get("chips", [])):
        ca = seg(t, 0.42 + i * 0.09, 0.66 + i * 0.09)
        if ca <= 0.02:
            continue
        cx += chip(d, (cx * S, (H - 132 + (1 - spring(ca)) * 26) * S), c, 34 * S,
                   PINE, mix(BG, (255, 255, 255), 0.8), min(1.0, ca * 2)) / S + 20

    d.rectangle([0, (H - 5) * S, int(W * t) * S, H * S], fill=fade(ACCENT, 0.55))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_pipeline(beat, fig, args, t):
    canvas = Image.new("RGB", (W, H), BG)
    fig = tint_to_bg(fig, BG)
    # this figure is squarer than the band, so it is height-limited:
    # every pixel of extra height buys width too
    im, fx, fy, fw, fh = stage_figure(fig, 296, H - 196, pad=70)
    n = len(PIPELINE_STEPS)
    assert len(beat["steps"]) == n, "one caption per highlighted stage"
    intro, outro = 0.03, 0.08
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
            for x0, y0, x1, y1 in PIPELINE_STEPS[i]:
                rd.rounded_rectangle([x0 * fw, y0 * fh, x1 * fw, y1 * fh],
                                     radius=12, fill=av)
        reveal = reveal.filter(ImageFilter.GaussianBlur(22))
    dim = Image.new("RGB", (fw, fh), BG)
    shown = Image.composite(im, Image.blend(im, dim, 0.80), reveal)
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
        for x0, y0, x1, y1 in PIPELINE_STEPS[k]:
            d.rounded_rectangle([(fx + x0 * fw) * S, (fy + y0 * fh) * S,
                                 (fx + x1 * fw) * S, (fy + y1 * fh) * S],
                                radius=10 * S, outline=fade(ACCENT, oa), width=5 * S)
    cap = PIPELINE_CLOSE if final else beat["steps"][k]
    ca = 1.0 if final else ease_out(seg(local, 0.04, 0.26))
    f = F(42 * S, True)
    tw = d.textlength(cap, font=f) / S
    bx, by = (W - tw) / 2 - 36, H - 168
    d.rounded_rectangle([bx * S, by * S, (bx + tw + 72) * S, (by + 84) * S],
                        radius=42 * S, fill=fade(mix(BG, (255, 255, 255), 0.78), ca))
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


def frame_table(beat, args, t):
    """A typeset table. Rows arrive one at a time, bars scale with the value,
    and the proposed row is held back until last and then marked."""
    spec = TABLES[beat["table"]]
    cols, rows, ours = spec["cols"], spec["rows"], spec["ours"]
    canvas = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS

    header(d, S, args, 1.0, small=True)
    ah = ease_out(seg(t, 0.0, 0.24))
    d.text(((86 - (1 - ah) * 34) * S, 162 * S), beat["head"], font=F(70 * S, True),
           fill=fade(PINE, ah))
    if beat.get("sub"):
        asub = ease_out(seg(t, 0.08, 0.34))
        d.text(((86 - (1 - asub) * 22) * S, 248 * S), beat["sub"],
               font=F(34 * S), fill=fade(MUTED, asub))
    d.rectangle([86 * S, 144 * S, (86 + ease_out(seg(t, 0.04, 0.4)) * 120) * S, 149 * S],
                fill=fade(ACCENT, 1))

    fh, fb = F(40 * S), F(40 * S, True)
    label_w = max(d.textlength(str(r[0]), font=fb) for r in rows) / S + 92
    num_w = 300
    table_w = label_w + num_w * (len(cols) - 1)
    x0 = (W - table_w) / 2
    # the band between the sub and the note has to hold header + rows
    top = 322
    row_h = min(84, int((H - 190 - top) / (len(rows) + 1)))
    body_h = row_h * (len(rows) + 1) + 26

    card_a = ease_out(seg(t, 0.03, 0.3))
    if card_a > 0.02:
        pad = 36
        plate = Image.new("RGB", (int(table_w + 2 * pad), int(body_h + 2 * pad)),
                          mix(BG, (255, 255, 255), 0.55))
        sh, ox, oy = shadow(plate.size, alpha=62)
        canvas.paste(sh, (int(x0 - pad) - ox, int(top - pad) - oy), sh)
        rc = round_corners(plate, 16)
        if card_a < 1:
            base = canvas.crop((int(x0 - pad), int(top - pad),
                                int(x0 - pad) + rc.width,
                                int(top - pad) + rc.height)).convert("RGBA")
            rc = Image.blend(base, rc, card_a)
        canvas.paste(rc, (int(x0 - pad), int(top - pad)), rc)

    hy = top + 10
    for i, (name, align, _) in enumerate(cols):
        cx = x0 + (0 if i == 0 else label_w + num_w * (i - 1))
        cw = label_w if i == 0 else num_w
        tw = d.textlength(name, font=fb) / S
        tx = cx + 18 if align == "l" else cx + cw - tw - 18
        d.text((tx * S, hy * S), name, font=fb, fill=fade(MUTED, card_a))
    d.rectangle([x0 * S, (hy + row_h - 12) * S, (x0 + table_w) * S,
                 (hy + row_h - 9) * S], fill=fade(MUTED, 0.45 * card_a))

    spans = {}
    for i in range(1, len(cols)):
        vals = [r[i] for r in rows if not isinstance(r[i], str)]
        if vals:
            spans[i] = (min(vals), max(vals))

    order = [i for i in range(len(rows)) if i != ours] + [ours]
    # Build finishes by mid-beat. The old schedule ran to t=0.76, so for a
    # quarter of the beat the card held two rows and read as broken -- which
    # at 376px, where the type is illegible anyway, is all you see.
    step = min(0.045, 0.30 / max(1, len(rows)))
    for slot, ri in enumerate(order):
        a = ease_out(seg(t, 0.08 + slot * step, 0.26 + slot * step))
        if a <= 0.02:
            continue
        is_ours = ri == ours
        y = hy + row_h + 6 + ri * row_h
        dx = (1 - a) * 26
        if is_ours:
            d.rounded_rectangle([(x0 - 14) * S, (y - 12) * S,
                                 (x0 + table_w + 14) * S, (y + row_h - 16) * S],
                                radius=12 * S, fill=fade(ACCENT, 0.12 * a))
        for i, (_, align, bar) in enumerate(cols):
            cx = x0 + (0 if i == 0 else label_w + num_w * (i - 1)) - dx
            cw = label_w if i == 0 else num_w
            val = rows[ri][i]
            dp = spec.get("dp", 2)
            txt = (str(val) if (i == 0 or isinstance(val, str))
                   else "%.*f%s" % (dp, val, spec["unit"]))
            f = fb if is_ours else fh
            if i and bar and i in spans and not isinstance(val, str):
                lo, hi = spans[i]
                frac = 0.12 + 0.88 * (val - lo) / max(1e-6, hi - lo)
                bw = (cw - 28) * frac * ease_out(seg(t, 0.14 + slot * step,
                                                     0.40 + slot * step))
                d.rounded_rectangle([(cx + cw - 20 - bw) * S, (y + 58) * S,
                                     (cx + cw - 20) * S, (y + 68) * S], radius=5 * S,
                                    fill=fade(ACCENT if is_ours else MUTED,
                                              (0.85 if is_ours else 0.38) * a))
            tw = d.textlength(txt, font=f) / S
            tx = cx + 18 if align == "l" else cx + cw - tw - 18
            d.text((tx * S, y * S), txt, font=f,
                   fill=fade(PINE if is_ours else INK, a))

    na = ease_out(seg(t, 0.54, 0.72))
    if spec.get("note") and na > 0.02:
        f = F(34 * S, True)
        tw = d.textlength(spec["note"], font=f) / S
        bx, by = (W - tw) / 2 - 32, H - 150
        d.rounded_rectangle([bx * S, by * S, (bx + tw + 64) * S, (by + 70) * S],
                            radius=35 * S, fill=fade(mix(BG, (255, 255, 255), 0.78), na))
        d.text(((bx + 32) * S, (by + 17) * S), spec["note"], font=f, fill=fade(PINE, na))

    d.rectangle([0, (H - 5) * S, int(W * t) * S, H * S], fill=fade(ACCENT, 0.55))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_end(beat, args, t):
    canvas = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    a = ease_out(seg(t, 0.02, 0.3))
    d.text((150 * S, 210 * S), "What carries the result", font=F(44 * S, True),
           fill=fade(PINE, a))
    d.rectangle([150 * S, 288 * S, (150 + a * 150) * S, 291 * S], fill=fade(ACCENT, a))
    for i, line in enumerate(beat["lines"]):
        la = ease_out(seg(t, 0.18 + i * 0.13, 0.5 + i * 0.13))
        if la <= 0.02:
            continue
        y = 370 + i * 92
        d.ellipse([152 * S, (y + 12) * S, 170 * S, (y + 30) * S], fill=fade(ACCENT, la))
        d.text((196 * S, y * S), line, font=F(28 * S), fill=fade(INK, la))
    fa = ease_out(seg(t, 0.55, 0.85))
    d.text((150 * S, 760 * S), args.source, font=F(26 * S, True), fill=fade(PINE, fa))
    d.text((150 * S, 806 * S), args.authors + "  .  " + args.affil,
           font=F(22 * S), fill=fade(MUTED, fa))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--title", default="Semi-Supervised Bangla Emotion")
    ap.add_argument("--title-lines", nargs="+",
                    default=["Semi-Supervised Emotion Classification",
                             "for Noisy Bangla Text"])
    ap.add_argument("--authors", default="Abdullah Al Shafi, Sumaiya Rahim Suma")
    ap.add_argument("--affil", default="Khulna University of Engineering & Technology")
    ap.add_argument("--venue", default="WIECON-ECE 2025")
    ap.add_argument("--source", default="EmoNoBa corpus, 22,698 comments, 12 domains")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every beat's duration")
    ap.add_argument("--full", action="store_true",
                    help="the long cut: 7 beats with title and closing cards")
    ap.add_argument("--palette", default="lilac", choices=list(PALETTES),
                    help="pastel family for this paper; give each paper its own")
    ap.add_argument("--themes", action="store_true",
                    help="print the contrast table for the palette and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check every TABLES value against the PDF text and exit")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default="bangla")
    args = ap.parse_args()

    build_palette(args.palette)
    if args.themes:
        sys.exit(0 if check_themes() else 1)
    if not args.pdf:
        sys.exit("give the paper PDF path")

    doc = fitz.open(args.pdf)
    folder = os.path.dirname(os.path.abspath(args.pdf))
    if args.grid:
        grid_pages(doc, folder)
        return
    if args.verify:
        sys.exit(0 if verify_tables(doc) else 1)
    verify_tables(doc)

    beats = BEATS if args.full else SHORT
    figs = {}
    for beat in beats:
        name = beat.get("figure")
        if not name or name in figs:
            continue
        figs[name] = trim_white(crop_figure(doc, FIGURES[name]))
        print("  %-14s %dx%d%s" % (name, figs[name].width, figs[name].height,
                                   "  (ground left alone: photographic)"
                                   if name in NO_TINT else "  (ground tinted)"))

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
        if b["kind"] == "table":
            return frame_table(b, args, t)
        if b["kind"] == "pipeline":
            return frame_pipeline(b, figs[b["figure"]], args, t)
        name = b["figure"]
        if name in NO_TINT:
            b = dict(b, _no_tint=True)
        return frame_figure(b, figs[name], args, t)

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

    # poster: the qualitative panel, rendered directly rather than sampled
    # from the stream. A fixed fraction of the clip lands wherever the beat
    # boundaries happen to be, and a transition frame posters as a double
    # exposure.
    pi = next((i for i, (b, _) in enumerate(plan)
               if b.get("table") == "sota"), 0)
    pb, pn = plan[pi]
    render(pb, int(pn * 0.86), pn, pi).save(stem + "_poster.jpg", quality=88,
                                            optimize=True, progressive=True)
    print(" " * 34, end="\r")
    for f in (stem + ".mp4", stem + "_poster.jpg"):
        if os.path.exists(f):
            print("  %-26s %9.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))


if __name__ == "__main__":
    main()
