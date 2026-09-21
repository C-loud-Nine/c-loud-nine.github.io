#!/usr/bin/env python3
r"""
make_dscbam_video.py

A paper video for "DSCBAM-Net: A Lightweight Neural Network with Grouped
Depthwise-Separable CBAM for Automated Brain Tumor Classification"
(IEEE EICT 2025), built straight from the paper PDF. Figures are cropped
from the source at high DPI, so nothing is a screenshot and nothing is
redrawn.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_dscbam_video.py "papers/dscbam.pdf"

Default is a short loop: the four classes, the architecture, the module,
the Grad-CAM. --full adds preprocessing, the benchmark table, the confusion
matrices, and title and closing cards.

    --grid    one PNG per page with a labelled coordinate overlay, so the
              FIGURES boxes can be read off instead of guessed
    --themes  WCAG contrast table for every beat palette

Every number on screen is quoted from the paper: Table I for the ablation,
Table II for the benchmark, Section III-B for the parameter counts.
"""

import argparse
import math
import os
import re
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
# Verified against the rendered pages; all ten land on their figure.
FIGURES = {
    "fig1_classes":   (0, 0.505, 0.740, 0.935, 0.850),
    "fig2_preproc":   (1, 0.510, 0.085, 0.935, 0.205),
    "fig3_arch":      (2, 0.090, 0.095, 0.925, 0.300),
    "fig4_module":    (2, 0.495, 0.365, 0.895, 0.545),
    "fig5_group":     (2, 0.495, 0.620, 0.905, 0.760),
    "fig6_training":  (3, 0.495, 0.695, 0.935, 0.850),
    "fig7_confusion": (4, 0.495, 0.380, 0.905, 0.550),
    "fig8_gradcam":   (4, 0.500, 0.583, 0.900, 0.880),
    "table1":         (4, 0.105, 0.505, 0.495, 0.725),
    "table2":         (4, 0.105, 0.712, 0.495, 0.930),
}

# the six stages of fig3, as fractions of the trimmed crop: the input slice,
# the two plain conv blocks, the three DSCBAM blocks, then the dense head
PIPELINE_STEPS = [
    (0.000, 0.330, 0.098, 0.720),   # 256x256x1 input slice
    (0.100, 0.150, 0.320, 0.740),   # C1 (32) and C2 (64)
    (0.316, 0.150, 0.432, 0.760),   # C3, 128 filters + DSCBAM
    (0.429, 0.150, 0.518, 0.760),   # C4, 256 filters + DSCBAM
    (0.513, 0.150, 0.628, 0.760),   # C5, 256 filters + DSCBAM
    (0.624, 0.010, 1.000, 0.990),   # flatten, dense head, softmax output
]

PIPELINE_CLOSE = "DSCBAM adds 5,393 parameters: 0.055% of the network"

# Diagrams and plots arrive on paper white, which sits on a tinted canvas as
# an obvious pasted rectangle. Recolouring their ground to the plate makes
# them read as drawn for the video. MRI slices and Grad-CAM overlays are
# photographs -- tinting those would alter the data, so they stay put.
TINT_GROUND = {"fig3_arch", "fig4_module", "fig5_group",
               "fig6_training", "fig7_confusion"}

# tables re-typeset from the PDF text layer instead of cropped as an image:
# (page, caption marker, first column header, number of numeric columns)
TABLES = {
    "table1": (4, "TABLE I", "Model", 4),
    "table2": (4, "TABLE II", "Method", 4),
}
TABLE_TITLES = {
    "table1": "Ablation study of component contributions",
    "table2": "Comparative benchmarking with state-of-the-art",
}
OURS = "DSCBAM-Net"

# ---------------------------------------------------------------- beats
BEATS = [
    dict(kind="title", seconds=4.0),
    dict(kind="figure", figure="fig1_classes", seconds=4.5,
         head="Four classes that look alike",
         sub="Subtle inter-class variation, artifacts and low contrast mask lesion boundaries",
         chips=["glioma", "meningioma", "pituitary", "no tumor"]),
    dict(kind="figure", figure="fig2_preproc", seconds=4.0,
         head="Crop to the anatomy, then equalise",
         sub="Contour-based extreme point detection, CLAHE, median filtering, 256x256 grayscale",
         chips=["+2.85% accuracy from preprocessing alone"]),
    dict(kind="pipeline", figure="fig3_arch", seconds=16.0,
         head="Five conv blocks, attention on the last three",
         sub="DSCBAM modules refine mid-level texture and high-level structure",
         steps=["A 256x256 grayscale slice enters",
                "Two blocks pick up edges and texture: 32 then 64 filters",
                "DSCBAM enters at block three, 128 filters",
                "Block four, 256 filters, attention again",
                "Block five, 256 filters, the last refinement",
                "Flatten into 512-512-256 dense layers and a softmax"]),
    dict(kind="figure", figure="fig4_module", seconds=6.0,
         head="Two cheap paths instead of one expensive one",
         sub="Grouped channel attention across four branches, then factorized 7x1 and 1x7 spatial attention",
         chips=["2,560 params vs 8,704 in CBAM", "same 7x7 receptive field"]),
    dict(kind="figure", figure="table2", seconds=5.5,
         head="Better accuracy than the transfer-learning baselines",
         sub="97.43% on the 4-class set, 95.96% on the 3-class set",
         chips=["beats ResNet50 at 95.25%", "beats VGG16 and DenseNet at 93%"]),
    dict(kind="figure", figure="fig7_confusion", seconds=4.5,
         head="Where it still confuses classes",
         sub="Glioma and meningioma remain the hard pair; pituitary is near-perfect",
         chips=["AUC 0.997 and 0.995"]),
    dict(kind="figure", figure="fig8_gradcam", seconds=5.0,
         head="The attention lands on the lesion",
         sub="Grad-CAM suppresses background and irrelevant anatomy",
         chips=["meningioma", "pituitary"]),
    dict(kind="end", seconds=5.0,
         lines=["DSCBAM adds 5,393 parameters: 0.055% of the network",
                "73% fewer parameters than the CBAM it replaces",
                "97.43 / 95.96 / 96.64% across three MRI datasets"]),
]


# The default cut: four beats, no title or end card, because the page around
# it already says whose paper this is. The four classes lead, since a row of
# MRI slices is the one frame that reads at thumbnail size; the architecture
# and the module explain it afterwards.
SHORT = [
    dict(kind="figure", figure="fig1_classes", seconds=2.4,
         head="Four tumour classes that look alike",
         sub="",
         chips=["MRI, 4-class"]),
    dict(kind="pipeline", figure="fig3_arch", seconds=9.0,
         head="Attention on the last three blocks",
         sub="DSCBAM refines mid-level texture and high-level structure",
         steps=["A 256x256 grayscale slice", "Edges and texture: 32 then 64 filters",
                "DSCBAM at block three", "Block four, 256 filters",
                "Block five, last refinement", "Dense head and softmax"]),
    dict(kind="figure", figure="fig4_module", seconds=2.8,
         head="Grouped channel plus factorized spatial attention",
         sub="",
         chips=["2,560 params vs 8,704"]),
    dict(kind="figure", figure="table2", seconds=3.4,
         head="Ahead of the transfer-learning baselines",
         sub="Typeset from the paper's own text layer, not cropped as an image",
         chips=["+2.18 over ResNet50", "+4.43 over VGG16"]),
    dict(kind="figure", figure="fig8_gradcam", seconds=2.6,
         head="97.43% accuracy, attention on the lesion",
         sub="",
         chips=["AUC 0.997", "+5,393 params total"]),
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


def pastel(c, amount=0.10, base=None):
    """A wash of c over base. Used for row banding: distinct enough to
    separate groups, faint enough not to fight the numbers."""
    base = CARD if base is None else base
    return tuple(int(b + (x - b) * amount) for x, b in zip(c, base))


def tint_ground(im, colour, thresh=238):
    """Recolour the near-white ground of a line drawing.

    Keyed on the channel MINIMUM, so a pale blue box (200, 220, 255) is
    untouched while paper white (255, 255, 255) moves fully to the plate
    colour, and the ramp between avoids a hard cut-out edge.
    """
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    t = np.clip((a.min(axis=2) - thresh) / (255.0 - thresh), 0, 1)[..., None]
    tgt = np.array(colour, dtype=np.float32)
    return Image.fromarray((a * (1 - t) + tgt * t).astype(np.uint8))


def parse_table(doc, page, marker, head0, ncols):
    """Read a table out of the PDF text layer.

    The numbers on screen are then the numbers in the paper, not a
    transcription of them, and not a raster of the typeset table.
    """
    lines = [l.strip() for l in doc[page].get_text().split("\n") if l.strip()]
    i = next(i for i, l in enumerate(lines) if marker in l)
    j = next(k for k in range(i, len(lines)) if lines[k] == head0)
    headers = lines[j:j + 1 + ncols]
    rows, k = [], j + 1 + ncols
    num = re.compile(r"^-?\d+(\.\d+)?$")
    while k < len(lines):
        lab = lines[k]
        vals = lines[k + 1:k + 1 + ncols]
        if len(vals) == ncols and all(num.match(v) for v in vals):
            rows.append((lab, vals))
            k += 1 + ncols
        elif lab.lower().startswith("dataset"):
            rows.append((lab, None))
            k += 1
        else:
            break
    return headers, rows


def render_table(headers, rows, title, width=2800):
    """Draw the table as type on a banded plate.

    Three pastel weights carry the structure: the dataset bands separate
    the blocks, the alternating wash keeps long rows trackable, and the
    accent band marks our own rows without needing a legend.
    """
    S = 2
    pad = 40
    title_h, head_h, row_h, grp_h = 124, 100, 82, 68
    body = sum(grp_h if v is None else row_h for _, v in rows)
    height = title_h + head_h + body + pad

    im = Image.new("RGB", (width * S, height * S), CARD)
    d = ImageDraw.Draw(im)
    rule = pastel(INK, 0.16)
    band_a, band_b = CARD, pastel(INK, 0.045)
    grp_bg = pastel(PINE, 0.13)
    ours_bg = pastel(ACCENT, 0.17)

    # column geometry: label column takes the slack, numbers share the rest
    ncol = len(headers) - 1
    num_w = int(width * 0.148)
    lab_w = width - pad * 2 - num_w * ncol
    xs = [pad] + [pad + lab_w + i * num_w for i in range(ncol + 1)]

    d.rectangle([0, 0, width * S, title_h * S], fill=pastel(PINE, 0.20))
    d.rectangle([0, (title_h - 5) * S, width * S, title_h * S], fill=ACCENT)
    ft = F(54 * S, True)
    d.text((pad * S, (title_h / 2 - 34) * S), title, font=ft, fill=PINE)

    y = title_h
    d.rectangle([0, y * S, width * S, (y + head_h) * S], fill=pastel(PINE, 0.07))
    fh = F(44 * S, True)
    d.text((xs[0] * S, (y + head_h / 2 - 28) * S), headers[0], font=fh, fill=PINE)
    for i, h in enumerate(headers[1:]):
        tw = d.textlength(h, font=fh) / S
        d.text(((xs[i + 1] + num_w - tw) * S, (y + head_h / 2 - 28) * S),
               h, font=fh, fill=PINE)
    y += head_h
    d.rectangle([0, (y - 3) * S, width * S, y * S], fill=rule)

    fr, fb = F(42 * S), F(42 * S, True)
    fg = F(38 * S, True)
    stripe = 0
    for lab, vals in rows:
        if vals is None:
            d.rectangle([0, y * S, width * S, (y + grp_h) * S], fill=grp_bg)
            d.text((xs[0] * S, (y + grp_h / 2 - 24) * S), lab, font=fg, fill=PINE)
            y += grp_h
            stripe = 0
            continue
        mine = lab.startswith(OURS)
        bg = ours_bg if mine else (band_a if stripe % 2 == 0 else band_b)
        d.rectangle([0, y * S, width * S, (y + row_h) * S], fill=bg)
        if mine:
            d.rectangle([0, y * S, 9 * S, (y + row_h) * S], fill=ACCENT)
        f = fb if mine else fr
        col = ACCENT if mine else INK
        d.text((xs[0] * S, (y + row_h / 2 - 27) * S), lab, font=f, fill=col)
        for i, v in enumerate(vals):
            tw = d.textlength(v, font=f) / S
            d.text(((xs[i + 1] + num_w - tw) * S, (y + row_h / 2 - 27) * S),
                   v, font=f, fill=col)
        y += row_h
        stripe += 1

    d.rectangle([0, (height - pad) * S, width * S, (height - pad + 3) * S], fill=rule)
    return im.resize((width, height), Image.LANCZOS)


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
    f = F(52 * S, True)
    for i, line in enumerate(args.title_lines):
        tw = d.textlength(line, font=f) / S
        d.text((((W - tw) / 2) * S, ((300 + i * 74) - dy) * S), line, font=f,
               fill=fade(PINE, a1))
    a2 = ease_out(seg(t, 0.25, 0.6))
    f2 = F(25 * S)
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


_tint_cache = {}


def plate_for(name):
    """Plate colour for this beat: a faint accent wash, shared by the card
    and by the figure's own ground so there is no seam between them."""
    return pastel(ACCENT, 0.07)


def tinted_figure(name, fig):
    key = (name, ACCENT, CARD)
    if key not in _tint_cache:
        _tint_cache[key] = tint_ground(fig, plate_for(name))
    return _tint_cache[key]


def frame_figure(beat, fig, args, t):
    canvas = Image.new("RGB", (W, H), BG)

    # The zoom grows the figure INTO its slot rather than past it. Scaling
    # up and centre-cropping loses ~2% off every edge, which clips the
    # sub-captions off figures that carry them.
    ZOOM = 1.045
    scale = (1.0 + (ZOOM - 1.0) * ease_io(t)) / ZOOM
    drift = int(10 * ease_io(t))
    base, fx, fy, fw, fh = stage_figure(fig, 318, H - 232)
    iw, ih = max(1, int(fw * scale)), max(1, int(fh * scale))
    zoomed = base.resize((iw, ih), Image.LANCZOS)

    a = ease_out(seg(t, 0.04, 0.40))
    if a > 0.02:
        pad = 18
        plate = Image.new("RGB", (fw + 2 * pad, fh + 2 * pad), beat.get("_plate", CARD))
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
    assert len(beat["steps"]) == n, "one caption per highlighted stage"
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
    d.text((150 * S, 210 * S), "What it costs", font=F(44 * S, True), fill=fade(PINE, a))
    d.rectangle([150 * S, 288 * S, (150 + a * 150) * S, 291 * S], fill=fade(ACCENT, a))
    for i, line in enumerate(beat["lines"]):
        la = ease_out(seg(t, 0.18 + i * 0.13, 0.5 + i * 0.13))
        if la <= 0.02:
            continue
        y = 370 + i * 92
        d.ellipse([152 * S, (y + 12) * S, 170 * S, (y + 30) * S], fill=fade(ACCENT, la))
        d.text((196 * S, y * S), line, font=F(29 * S), fill=fade(INK, la))
    fa = ease_out(seg(t, 0.55, 0.85))
    d.text((150 * S, 760 * S), args.url, font=F(26 * S, True), fill=fade(PINE, fa))
    d.text((150 * S, 806 * S), args.authors + "  .  " + args.affil,
           font=F(22 * S), fill=fade(MUTED, fa))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--title", default="DSCBAM-Net")
    ap.add_argument("--title-lines", nargs="+",
                    default=["DSCBAM-Net",
                             "Grouped depthwise-separable attention for brain tumour MRI"])
    ap.add_argument("--authors",
                    default="Abdullah Al Shafi, Sumaiya Rahim Suma, Sk. Md. Masudul Ahsan")
    ap.add_argument("--affil", default="Khulna University of Engineering & Technology")
    ap.add_argument("--venue", default="IEEE EICT 2025")
    ap.add_argument("--url", default="ieeexplore.ieee.org/document/11355571")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every beat's duration")
    ap.add_argument("--full", action="store_true",
                    help="the long cut: 9 beats with title and closing cards")
    ap.add_argument("--themes", action="store_true",
                    help="print the contrast table for every theme and exit")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default="dscbam")
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
    figs, tinted = {}, {}
    for j, beat in enumerate(beats):
        name = beat.get("figure")
        if not name or name in figs:
            continue
        if name in TABLES:
            set_theme(j)
            heads, rows = parse_table(doc, *TABLES[name])
            figs[name] = render_table(heads, rows, TABLE_TITLES[name])
            print("  %-16s %dx%d  (%d rows, typeset from the text layer)"
                  % (name, figs[name].width, figs[name].height, len(rows)))
            continue
        figs[name] = trim_white(crop_figure(doc, FIGURES[name]))
        if name in TINT_GROUND:
            tinted[name] = True
        print("  %-16s %dx%d%s" % (name, figs[name].width, figs[name].height,
                                   "  (ground tinted)" if name in tinted else ""))

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
        name = b["figure"]
        fg = figs[name]
        if name in tinted:
            fg = tinted_figure(name, fg)
            b = dict(b, _plate=plate_for(name))
        elif name in TABLES:
            b = dict(b, _plate=CARD)
        return frame_figure(b, fg, args, t)

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
