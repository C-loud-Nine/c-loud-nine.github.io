#!/usr/bin/env python3
r"""
make_gamp_video.py

A paper video for "Closing the Null Space: Guidance-Aware Quantization for
Classifier-Free Diffusion" (IEEE HPEC 2026), built straight from the paper
PDF. Figures are cropped from the source at high DPI, so nothing is a
screenshot and nothing is redrawn.

    pip install pymupdf python-pptx pillow numpy imageio imageio-ffmpeg
    python make_gamp_video.py "gamp/2607.08241v1.pdf" --pptx "gamp/HPEC_GAMP.pptx"

Beats (--full)
  1  title card
  2  the CFG tax             Fig. 2
  3  the INT8 deployment gap Fig. 3
  4  the branch-drift trap   Fig. 4     <- the paper's punchline
  5  GAMP pipeline           Fig. 1     (revealed step by step)
  6  Pareto result           Fig. 5
  7  takeaways card

Default is the short cut: three beats, ~8s, no title or end card, because
the page around it already says whose paper this is.

Adjusting crops
  --grid writes one PNG per page with a labelled coordinate overlay. Read
  the fractions off it and edit FIGURES. Boxes are (page, x0, y0, x1, y1)
  as fractions of the page.

Every number on screen is quoted from the paper: Table III for the row
values, Section IV-C for the 147x slowdown and 192 memcpy nodes, Table I
for the 1.99x CFG overhead.
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
# lifted from style.css so the clip sits on the page rather than on top of it
BG = (251, 246, 236)            # --bg          warm paper
INK = (52, 48, 42)              # --fg
MUTED = (138, 131, 119)         # --muted
ACCENT = (161, 61, 36)          # --accent      terracotta
PINE = (27, 58, 47)             # --heading     deep pine
SOFT = (230, 217, 196)          # --accent-soft
CARD = (255, 253, 250)

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


# ---- figures taken from the slide deck instead of the PDF, when --pptx is
# given. The deck's pipeline is a standalone PNG at 1650x385, which beats
# cropping the PDF figure away from its caption block.
FROM_DECK = {
    "fig1_pipeline": (10, 1),
}

# ------------------------------------------------- figure crops (page, box)
# Verified against the rendered pages; all six land on their figure.
FIGURES = {
    "fig1_pipeline": (3, 0.060, 0.048, 0.925, 0.190),
    "fig2_cfgtax":   (3, 0.495, 0.275, 0.930, 0.515),
    "fig3_int8":     (4, 0.100, 0.052, 0.500, 0.300),
    "fig4_trap":     (5, 0.100, 0.052, 0.500, 0.292),
    "fig5_pareto":   (5, 0.070, 0.352, 0.490, 0.586),
    "table3":        (4, 0.495, 0.060, 0.920, 0.345),
}

# six GAMP stages as fractions of the cropped fig1, for the staged reveal
PIPELINE_STEPS = [
    (0.015, 0.22, 0.180, 0.97),
    (0.183, 0.22, 0.335, 0.97),
    (0.340, 0.22, 0.497, 0.97),
    (0.502, 0.22, 0.660, 0.97),
    (0.665, 0.22, 0.822, 0.97),
    (0.828, 0.22, 1.000, 0.72),
]

# ---------------------------------------------------------------- beats
BEATS = [
    dict(kind="title", seconds=4.0),
    dict(kind="figure", figure="fig2_cfgtax", seconds=5.0,
         head="CFG runs the network twice at every step",
         sub="Measured on a T4 across batch sizes and float formats",
         chips=["1.99x latency tax", "invisible to BOPs", "invisible to parameter counts"]),
    dict(kind="figure", figure="fig3_int8", seconds=5.0,
         head="INT8 that should be 16x faster is 147x slower",
         sub="Not an arithmetic limit: 192 operator-fallback memcpy nodes",
         chips=["BOPs predict 16x", "measured 0.007x", "software-stack problem"]),
    dict(kind="figure", figure="fig4_trap", seconds=6.0,
         head="The branch-drift trap",
         sub="Error common to both branches cancels in the gap, so nothing constrains it",
         chips=["rho = 1.004", "cos(delta) = 0.882", "FID 334"]),
    dict(kind="pipeline", figure="fig1_pipeline", seconds=17.0,
         head="GAMP: calibrate on the guided prediction",
         sub="The guided output is built from the unconditional branch, so the drift is inside what it measures",
         steps=["Dual-branch calibration captures both activation ranges",
                "Cache the full-precision guided output once",
                "Per-layer sensitivity from guided-output degradation",
                "Greedy knapsack promotes A4 to A8 by sensitivity per BOP",
                "Threshold calibration on the guided prediction closes the trap",
                "Mixed-precision model: W4, non-uniform activations"]),
    dict(kind="figure", figure="fig5_pareto", seconds=5.0,
         head="49% better FID at matched average precision",
         sub="Table III: GAMP's two operating points",
         # b6 and b5 are DIFFERENT rows. Unlabelled adjacent chips read as
         # one configuration, claiming FID 39.40 and -13% BOPs together --
         # b6 is only 4.3% below W4A8 (31.3 vs 32.7 G BOPs).
         chips=["6.03 bits: FID 39.40", "5.01 bits: FID 40.01, 13% fewer BOPs"]),
    dict(kind="end", seconds=4.5,
         lines=["Report guided-step BOPs, not single-pass",
                "Validate PTQ with guided-prediction error, not gap diagnostics",
                "Allocate activation bits by guided-output sensitivity"]),
]


# The default cut. Eight seconds, three beats, no title or end card: the
# page around it already says whose paper this is. Result-first because the
# paradox is the hook -- near-perfect diagnostics, FID 334.
SHORT = [
    dict(kind="figure", figure="fig4_trap", seconds=3.2,
         head="Perfect gap fidelity, corrupted samples",
         sub="Gap-only calibration: rho 1.004, cos 0.882, FID 334",
         chips=["the branch-drift trap"]),
    dict(kind="pipeline", figure="fig1_pipeline", seconds=10.5,
         head="GAMP calibrates on the guided prediction",
         sub="Activation bits allocated by guided-output sensitivity",
         steps=["Dual-branch calibration", "Cache the FP32 guided output",
                "Per-layer guided-output sensitivity", "Greedy bit allocation",
                "Threshold calibration closes the null space",
                "W4, non-uniform activations"]),
    dict(kind="figure", figure="fig5_pareto", seconds=2.4,
         head="49% better FID at matched precision",
         sub="",
         chips=["6.03 bits: FID 39.40", "5.01 bits: 13% fewer BOPs"]),
]


def step_schedule(steps):
    """Screen time per pipeline step, proportional to its caption length.

    Uniform timing gave "Dual-branch calibration" and "Threshold calibration
    closes the null space" the same hold, so the longest label always ran
    fastest -- 7.6x faster than it can be read, in the short cut.
    """
    w = [max(2.0, len(s.split())) for s in steps]
    tot = sum(w)
    out, acc = [], 0.0
    for x in w:
        out.append((acc / tot, (acc + x) / tot))
        acc += x
    return out


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_io(t):
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def seg(t, a, b):
    return 0.0 if b <= a else max(0.0, min(1.0, (t - a) / (b - a)))


def fade(c, a):
    return tuple(c) + (int(max(0, min(255, a * 255))),)


def flatten(im):
    """Composite onto the page colour. .convert('RGB') on an RGBA image
    discards alpha instead of compositing it, so transparent regions come
    through as whatever RGB happened to be stored under them."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        out = Image.new("RGB", im.size, BG)
        out.paste(im, mask=im.split()[-1])
        return out
    return im.convert("RGB")


def figures_from_deck(path):
    try:
        from pptx import Presentation
    except ImportError:
        print("  --pptx needs: pip install python-pptx (ignoring the deck)")
        return {}
    import io
    want = {(sl, n): key for key, (sl, n) in FROM_DECK.items()}
    out = {}
    for i, slide in enumerate(Presentation(path).slides, 1):
        n = 0
        for sh in slide.shapes:
            if sh.shape_type is not None and "PICTURE" in str(sh.shape_type):
                n += 1
                key = want.get((i, n))
                if key:
                    out[key] = flatten(Image.open(io.BytesIO(sh.image.blob)))
                    print("  %-16s from deck slide %d: %dx%d"
                          % (key, i, out[key].width, out[key].height))
    return out


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


def shadow(size, radius=16, blur=26, alpha=52):
    """Memoised: card geometry never changes, and blurring it once per frame
    was the single largest cost in the render."""
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
    d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=fade(bg, a),
                        outline=fade(SOFT, a), width=max(1, int(size * 0.045)))
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
    im, fx, fy, fw, fh = stage_figure(fig, 318, H - 232)
    a = ease_out(seg(t, 0.06, 0.42))
    if a > 0.02:
        card = round_corners(im, 14)
        sh, ox, oy = shadow(im.size)
        dy = int((1 - a) * 22)
        if a > 0.3:
            canvas.paste(sh, (fx - ox, fy + dy - oy), sh)
        if a < 1.0:
            base = canvas.crop((fx, fy + dy, fx + fw, fy + dy + fh)).convert("RGBA")
            card = Image.blend(base, card, a)
        canvas.paste(card, (fx, fy + dy), card)

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    header(d, S, args, 1.0, small=True)
    ah = ease_out(seg(t, 0.02, 0.3))
    d.text((86 * S, 162 * S), beat["head"], font=F(70 * S, True), fill=fade(PINE, ah))
    d.text((86 * S, 248 * S), beat.get("sub", ""), font=F(34 * S), fill=fade(MUTED, ah))
    cx = 86
    for i, c in enumerate(beat.get("chips", [])):
        ca = ease_out(seg(t, 0.45 + i * 0.1, 0.72 + i * 0.1))
        if ca <= 0.02:
            continue
        cx += chip(d, (cx * S, (H - 132 + (1 - ca) * 8) * S), c, 34 * S,
                   PINE, CARD, ca) / S + 16
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def frame_pipeline(beat, fig, args, t):
    canvas = Image.new("RGB", (W, H), BG)
    im, fx, fy, fw, fh = stage_figure(fig, 296, H - 196, pad=70)
    n = len(PIPELINE_STEPS)
    assert len(beat["steps"]) == n, "one caption per highlighted step"
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
    cap = beat["steps"][k] if not final else "Inference cost equals standard mixed-precision PTQ"
    ca = 1.0 if final else ease_out(seg(local, 0.04, 0.26))
    f = F(42 * S, True)
    tw = d.textlength(cap, font=f) / S
    bx, by = (W - tw) / 2 - 36, H - 168
    d.rounded_rectangle([bx * S, by * S, (bx + tw + 72) * S, (by + 84) * S],
                        radius=42 * S, fill=fade(CARD, ca),
                        outline=fade(SOFT, ca), width=2 * S)
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
    d.text((150 * S, 210 * S), "For practitioners", font=F(44 * S, True), fill=fade(PINE, a))
    d.rectangle([150 * S, 288 * S, (150 + a * 150) * S, 291 * S], fill=fade(ACCENT, a))
    for i, line in enumerate(beat["lines"]):
        la = ease_out(seg(t, 0.18 + i * 0.13, 0.5 + i * 0.13))
        if la <= 0.02:
            continue
        y = 370 + i * 92
        d.ellipse([152 * S, (y + 12) * S, 170 * S, (y + 30) * S], fill=fade(ACCENT, la))
        d.text((196 * S, y * S), line, font=F(30 * S), fill=fade(INK, la))
    fa = ease_out(seg(t, 0.55, 0.85))
    d.text((150 * S, 760 * S), args.arxiv, font=F(26 * S, True), fill=fade(PINE, fa))
    d.text((150 * S, 806 * S), args.authors + "  .  " + args.affil,
           font=F(22 * S), fill=fade(MUTED, fa))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--title", default="Closing the Null Space")
    ap.add_argument("--title-lines", nargs="+",
                    default=["Closing the Null Space:",
                             "Guidance-Aware Quantization for Classifier-Free Diffusion"])
    ap.add_argument("--authors", default="Abdullah Al Shafi, Sumaiya Rahim Suma")
    ap.add_argument("--affil", default="Khulna University of Engineering & Technology")
    ap.add_argument("--venue", default="IEEE HPEC 2026 (Oral)")
    ap.add_argument("--arxiv", default="arXiv:2607.08241")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every beat's duration")
    ap.add_argument("--full", action="store_true",
                    help="the long cut: 7 beats with title and takeaways cards")
    ap.add_argument("--pptx", help="HPEC deck; supplies a cleaner pipeline figure")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default="gamp")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    folder = os.path.dirname(os.path.abspath(args.pdf))
    if args.grid:
        grid_pages(doc, folder)
        return

    beats = BEATS if args.full else SHORT
    figs = figures_from_deck(args.pptx) if args.pptx else {}
    for beat in beats:
        name = beat.get("figure")
        if not name or name in figs:
            continue
        if name not in FIGURES:
            sys.exit("%s is only available from the deck; pass --pptx" % name)
        figs[name] = crop_figure(doc, FIGURES[name])
        print("  %-16s %dx%d (from PDF)" % (name, figs[name].width, figs[name].height))
    figs = {k: trim_white(v) for k, v in figs.items()}

    xf = max(4, int(0.3 * args.fps))
    plan = [(b, max(xf * 2, int(b["seconds"] * args.scale * args.fps))) for b in beats]
    # per-beat fade length: never let a crossfade eat more than a quarter of
    # a beat, or a short beat has nothing legible left
    xs = [max(3, min(xf, n // 4)) for _, n in plan]
    L = len(plan)
    count = sum(n for _, n in plan) - sum(xs)

    def render(b, i, n):
        t = i / n
        if b["kind"] == "title":
            return frame_title(args, t)
        if b["kind"] == "end":
            return frame_end(b, args, t)
        if b["kind"] == "pipeline":
            return frame_pipeline(b, figs[b["figure"]], args, t)
        return frame_figure(b, figs[b["figure"]], args, t)

    def stream():
        for j, (b, n) in enumerate(plan):
            nb, nn = plan[(j + 1) % L]
            # body starts at the PREVIOUS beat's fade length: those opening
            # frames already played inside the crossfade coming in, and
            # replaying them stutters the beat
            for i in range(xs[(j - 1) % L], n - xs[j]):
                yield render(b, i, n)
            for k in range(xs[j]):
                a = ease_io((k + 1) / (xs[j] + 1))
                yield Image.blend(render(b, n - xs[j] + k, n), render(nb, k, nn), a)

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
    # and a crossfade frame posters as a double exposure
    pb, pn = plan[0]
    render(pb, int(pn * 0.88), pn).save(stem + "_poster.jpg", quality=92,
                                        optimize=True, progressive=True)
    print(" " * 34, end="\r")
    for f in (stem + ".mp4", stem + "_poster.jpg"):
        if os.path.exists(f):
            print("  %-26s %9.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))


if __name__ == "__main__":
    main()
