#!/usr/bin/env python3
r"""
make_onehealth_video.py

An animated graphical abstract for OneHealth+: the verification-first
request lifecycle annotated stage by stage, the clinician verification
interface, and the fail-safe handling of non-MRI input.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_onehealth_video.py path/to/onehealth.pdf

    --grid    write a coordinate overlay for Figure 3 and exit, so the STAGES
              boxes below can be read off instead of guessed
    --themes  WCAG contrast table for the palette

Three beats. A software paper has no benchmark table worth typesetting, so
the second and third beats are the interface itself.

Type scale, canvas geometry and pacing follow make_dash_abstract.py so the
clips sit together on the same page.

Nothing here is a screenshot of a table: the only rasters are the paper's
own figures, cropped at high DPI from the source PDF.
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- palette
BG = (239, 243, 247)            # clinical blue-grey, distinct from the siblings
INK = (24, 32, 42)
MUTED = (96, 108, 122)
ACCENT = (36, 96, 168)          # clinical blue
PINE = (22, 52, 92)
SOFT = (212, 222, 234)
CARD = (253, 254, 255)

W, H = 1600, 900                # 16:9, matches .pub_thumb
SS = 2                          # overlay supersampling: crisp edges
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


def fit(d, text, size, bold, limit, S=1, floor=0.6):
    """Largest font at or below `size` whose rendering fits `limit`."""
    lo = max(10, int(size * floor))
    for px in range(int(size), lo - 1, -1):
        f = F(px * S, bold)
        if d.textlength(text, font=f) / S <= limit:
            return f, px
    return F(lo * S, bold), lo


def _lum(c):
    c = [x / 255 for x in c]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    l1, l2 = sorted((_lum(a), _lum(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def check_theme():
    """WCAG ratios against both grounds the type sits on. Body text wants
    >= 4.5, large headings >= 3.0."""
    ok = True
    for name, c, need in (("ink", INK, 4.5), ("heading", PINE, 3.0),
                          ("muted", MUTED, 4.5), ("accent", ACCENT, 3.0)):
        a, b = contrast(c, BG), contrast(c, CARD)
        ok &= min(a, b) >= need
        print("  %-9s on BG %5.2f  on CARD %5.2f  %s"
              % (name, a, b, "ok" if min(a, b) >= need else "FAIL (needs %.1f)" % need))
    print("  %-9s on PINE %5.2f" % ("badge", contrast((255, 255, 255), PINE)))
    print("all pass" if ok else "SOME FAIL")
    return ok


# -------------------------------------- stages for OneHealth+ Figure 3
# Read off Figure 3 with --grid, then checked against the PDF text layer:
# every box below contains the label it claims to. Figure 3 has eight
# labelled boxes; Classify and Explain are lit together because one caption
# covers both. Boxes include each module's caption text, not just its
# outline -- a highlight that cuts the words off under it looks like a
# misregistration.
#
# Order follows the narrative, not the page: the reject branch is lit right
# after the validation step that triggers it, then the walk returns to the
# main path.
STAGES = [
    ("A patient uploads an MRI slice",
     [(0.005, 0.020, 0.205, 0.285)]),
    ("It is screened as a genuine MRI first",
     [(0.270, 0.020, 0.470, 0.285)]),
    ("Non-MRI input is rejected, disclaimed and logged",
     [(0.225, 0.300, 0.800, 0.585)]),
    ("DSCBAM-Net classifies it, Grad-CAM shows where it looked",
     [(0.530, 0.020, 0.990, 0.285)]),
    ("The prediction is withheld, pending review",
     [(0.005, 0.715, 0.205, 0.985)]),
    ("A clinician records a verdict, with a timestamp",
     [(0.270, 0.715, 0.600, 0.985)]),
    ("Only then is a verified report issued",
     [(0.665, 0.715, 0.990, 0.985)]),
]


def stage_schedule(stages):
    """Screen time per stage, proportional to caption length, so the longest
    line does not end up with the shortest hold."""
    w = [max(3.0, len(c.split())) for c, _ in stages]
    tot = sum(w)
    out, acc = [], 0.0
    for x in w:
        out.append((acc / tot, (acc + x) / tot))
        acc += x
    return out


def report_pacing(seconds, intro=0.10, outro=0.12):
    """Print the reading rate per stage. Captions that scroll past faster
    than about three words a second cannot actually be read at thumbnail
    size, which is the failure mode this catches."""
    body = seconds * (1.0 - intro - outro)
    worst = 0.0
    print("  caption pacing over %.1fs of walk:" % body)
    for (cap, _), (a0, a1) in zip(STAGES, stage_schedule(STAGES)):
        s = (a1 - a0) * body
        rate = len(cap.split()) / s
        worst = max(worst, rate)
        print("    %5.2fs  %4.1f w/s  %s" % (s, rate, cap))
    print("  worst %.1f words/sec %s" % (worst, "ok" if worst <= 3.4 else "TOO FAST"))
    return worst


# ---------------------------------------------------------------- figures
# (page index, x0, y0, x1, y1) as fractions of the page. Verified against
# the embedded image bounding boxes reported by PyMuPDF.
#
# "visual" stops at 0.398 rather than the figure's true bottom edge of
# 0.413: the paper's own screenshot is cut mid-way through a row of verdict
# buttons, and a half-height control reads as a rendering fault in motion.
#
# Its x bounds are inset to 0.2152-0.7724 because the screenshot captured
# the web app's grey page background either side of the card: 149px left and
# 183px right, which trim_white leaves alone because grey is not white. On a
# card plate those bands read as careless cropping.
FIGURES = {
    "arch":     (6, 0.150, 0.130, 0.850, 0.318),    # Fig. 3, request lifecycle
    "visual":   (8, 0.2152, 0.150, 0.7724, 0.398),  # Fig. 5, verification UI
    "failsafe": (9, 0.150, 0.150, 0.850, 0.318),    # Fig. 6, valid vs non-MRI
}


# ---------------------------------------------------------------- easing
def ease_io(t):
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def seg(t, a, b):
    return 0.0 if b <= a else max(0.0, min(1.0, (t - a) / (b - a)))


def fade(c, a):
    return tuple(c) + (int(max(0, min(255, a * 255))),)


# ------------------------------------------------------------------ io
def _fitz():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            sys.exit("pip install pymupdf")


def crop_figure(doc, spec, dpi=DPI):
    page, x0, y0, x1, y1 = spec
    pix = doc[page].get_pixmap(dpi=dpi, alpha=False)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return im.crop((int(x0 * im.width), int(y0 * im.height),
                    int(x1 * im.width), int(y1 * im.height)))


def trim_white(im, thresh=248, pad=8):
    """Drop the white page margin the crop inevitably includes."""
    a = np.asarray(im.convert("L"))
    mask = a < thresh
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))


_tint_cache = {}


def tint(im, bg, cut=243):
    """Repaint the figure's white paper ground in the page colour. Left
    white, every crop announces itself as a screenshot pasted on the slide.
    Only near-neutral bright pixels move, so plotted data, photographs and
    black text keep their values."""
    key = (id(im), bg, cut)
    if key in _tint_cache:
        return _tint_cache[key]
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    mx, mn = a.max(2), a.min(2)
    neutral = (mx - mn) < 14
    w = (np.clip((mx - cut) / (255.0 - cut), 0, 1) * neutral)[..., None]
    out = Image.fromarray((a * (1 - w) + np.asarray(bg, np.float32) * w).astype(np.uint8))
    _tint_cache[key] = out
    return out


def grid_overlay(fig, out):
    im = fig.copy().convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    w, h = im.size
    for i in range(1, 20):
        x = w * i / 20
        d.line([(x, 0), (x, h)], fill=(255, 0, 0) if i % 5 == 0 else (255, 180, 180),
               width=2 if i % 5 == 0 else 1)
        d.text((x + 3, 4), "%.2f" % (i / 20), fill=(200, 0, 0), font=F(18, True))
    for j in range(1, 10):
        y = h * j / 10
        d.line([(0, y), (w, y)], fill=(0, 0, 255) if j % 5 == 0 else (180, 180, 255),
               width=2 if j % 5 == 0 else 1)
        d.text((5, y + 3), "%.1f" % (j / 10), fill=(0, 0, 200), font=F(18, True))
    im.save(out)
    print("coordinate grid written to", out)


# ---------------------------------------------------------------- helpers
_shadow_cache = {}


def shadow(size, radius=14, blur=22, alpha=52):
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


def header(d, S, args, a=1.0, dy=0.0):
    """Paper title block, identical on every beat. 62/28 with a 130px rule,
    matching make_dash_abstract.py."""
    ft, _ = fit(d, args.title, 62, True, W - 640, S)
    d.text((72 * S, (36 - dy) * S), args.title, font=ft, fill=fade(PINE, a))
    fs, _ = fit(d, args.subtitle, 28, False, W - 200, S)
    d.text((72 * S, (108 - dy) * S), args.subtitle, font=fs, fill=fade(MUTED, a))


def caption_plate(d, S, text, a):
    """Bottom caption, 36 bold on a card plate, as in the DASH figure walk."""
    if a <= 0.01:
        return
    f, _ = fit(d, text, 36, True, W - 240, S)
    tw = d.textlength(text, font=f) / S
    bx, by = (W - tw) / 2 - 34, H - 126
    d.rounded_rectangle([bx * S, by * S, (bx + tw + 68) * S, (by + 66) * S],
                        radius=33 * S, fill=fade(CARD, a), outline=fade(SOFT, a), width=2 * S)
    d.text(((bx + 34) * S, (by + 16) * S), text, font=f, fill=fade(PINE, a))


def dots(d, S, n, active, a=1.0):
    for i in range(n):
        r = 5 * S
        x = (W / 2 - (n - 1) * 11 + i * 22) * S
        y = (H - 48) * S
        on = i <= active
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=fade(ACCENT if i == active else (PINE if on else MUTED),
                            (1.0 if on else 0.28) * a))


def badge(d, S, text, a):
    """Top-right pill, 34 bold, h=68 -- the DASH dataset badge."""
    f, _ = fit(d, text, 34, True, 460, S)
    tw = d.textlength(text, font=f) / S
    cw = tw + 58
    d.rounded_rectangle([(W - 72 - cw) * S, 46 * S, (W - 72) * S, 114 * S],
                        radius=34 * S, fill=fade(PINE, a))
    d.text(((W - 72 - cw + 29) * S, 62 * S), text, font=f, fill=fade((255, 255, 255), a))


# ---------------------------------------------------------------- frames
def figure_frame(fig, fx, fy, fw, fh, t, args):
    """The lifecycle walk. Earlier stages stay lit; the active one is
    spotlit and outlined."""
    canvas = Image.new("RGB", (W, H), BG)
    n = len(STAGES)
    intro, outro = 0.10, 0.12
    body = 1.0 - intro - outro
    sched = stage_schedule(STAGES)
    u = (t - intro) / body if body > 0 else 0.0
    k, local = 0, 0.0
    if t >= intro:
        for i, (a0, a1) in enumerate(sched):
            if u < a1 or i == n - 1:
                k, local = i, min(1.0, max(0.0, (u - a0) / (a1 - a0)))
                break
    final = t > intro + body

    reveal = Image.new("L", (fw, fh), 0)
    rd = ImageDraw.Draw(reveal)
    if final:
        reveal = Image.new("L", (fw, fh), 255)
    elif t >= intro:
        for i in range(k + 1):
            a = 255 if i < k else int(255 * ease_out(seg(local, 0.0, 0.45)))
            for (x0, y0, x1, y1) in STAGES[i][1]:
                rd.rounded_rectangle([x0 * fw, y0 * fh, x1 * fw, y1 * fh], radius=14, fill=a)
        reveal = reveal.filter(ImageFilter.GaussianBlur(26))

    dim = Image.new("RGB", (fw, fh), BG)
    shown = Image.composite(fig, Image.blend(fig, dim, args.dim), reveal)
    ia = ease_out(seg(t, 0.0, intro))
    if ia < 1.0:
        shown = Image.blend(Image.new("RGB", (fw, fh), BG), shown, ia)
    canvas.paste(shown, (fx, fy))

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    ti = ease_out(seg(t, 0.0, 0.08))
    header(d, S, args, ti)
    d.rectangle([72 * S, 152 * S, (72 + ease_out(seg(t, 0.02, 0.12)) * 130) * S, 156 * S],
                fill=fade(ACCENT, ti))

    if not final and t >= intro:
        oa = ease_out(seg(local, 0.0, 0.4)) * (1 - ease_out(seg(local, 0.90, 1.0)))
        for (x0, y0, x1, y1) in STAGES[k][1]:
            d.rounded_rectangle([(fx + x0 * fw) * S, (fy + y0 * fh) * S,
                                 (fx + x1 * fw) * S, (fy + y1 * fh) * S],
                                radius=10 * S, outline=fade(ACCENT, oa), width=4 * S)

    caption_plate(d, S, args.final_caption if final else STAGES[k][0],
                  1.0 if final else ease_out(seg(local, 0.04, 0.26)))
    dots(d, S, n, k if not final else n - 1)
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def visual_frame(halves, t, args, badge_text="", caption=""):
    """One or more figure panels, side by side on card plates."""
    canvas = Image.new("RGB", (W, H), BG)
    gap, pad = 40, 16
    n_panels = max(1, len(halves))
    avail_w = (W - 150 - gap * (n_panels - 1)) // n_panels
    # the band between the header rule and the caption plate. Figure 5 is
    # height-limited in it, so every pixel reclaimed here is one the
    # interface actually gains.
    band_top, band_bottom = 180, H - 150
    avail_h = band_bottom - band_top - 24
    z = 1.0 + 0.03 * ease_io(t)
    scaled = []
    for hf in halves:
        # margin before the slow zoom: the zoom crops ~1.5% from each edge,
        # which clipped text sitting flush against the figure boundary
        m_ = max(8, int(0.03 * max(hf.size)))
        padded = Image.new("RGB", (hf.width + 2 * m_, hf.height + 2 * m_), CARD)
        padded.paste(hf, (m_, m_))
        hf = padded
        s_ = min(avail_w / hf.width, avail_h / hf.height)
        fw, fh = int(hf.width * s_), int(hf.height * s_)
        zw, zh = int(fw * z), int(fh * z)
        im = hf.resize((zw, zh), Image.LANCZOS).crop(
            ((zw - fw) // 2, (zh - fh) // 2, (zw - fw) // 2 + fw, (zh - fh) // 2 + fh))
        scaled.append(im)
    # each panel carries its own plate padding; gaps only sit BETWEEN panels.
    # Counting a gap unconditionally left a single panel 36px left of centre.
    tw_ = sum(i.width + 2 * pad for i in scaled) + gap * (n_panels - 1)
    x = (W - tw_) // 2
    # centre in the band rather than pinning to its top, so a wide panel
    # does not leave all its slack underneath
    ph_ = max(i.height for i in scaled) + 2 * pad
    fy = band_top + max(0, (band_bottom - band_top - ph_) // 2)

    for k, im in enumerate(scaled):
        a = ease_out(seg(t, 0.04 + k * 0.08, 0.40 + k * 0.08))
        if a <= 0.02:
            x += im.width + gap + 2 * pad
            continue
        plate = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), CARD)
        plate.paste(im, (pad, pad))
        card = round_corners(plate, 14)
        sh, ox, oy = shadow(plate.size)
        dy = int((1 - a) * 24)
        canvas.paste(sh, (x - ox, fy + dy - oy), sh)
        if a < 1.0:
            base = canvas.crop((x, fy + dy, x + card.width,
                                fy + dy + card.height)).convert("RGBA")
            card = Image.blend(base, card, a)
        canvas.paste(card, (x, fy + dy), card)
        x += im.width + gap + 2 * pad

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    header(d, S, args, 1.0)
    d.rectangle([72 * S, 152 * S, (72 + 130) * S, 156 * S], fill=fade(ACCENT, 1))
    badge(d, S, badge_text, ease_out(seg(t, 0.02, 0.26)))
    caption_plate(d, S, caption, ease_out(seg(t, 0.30, 0.52)))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


# ---------------------------------------------------------------- build
def build(args):
    doc = _fitz().open(args.pdf)
    arch = tint(trim_white(crop_figure(doc, FIGURES["arch"])), BG)
    visual = [tint(trim_white(crop_figure(doc, FIGURES["visual"])), CARD)]
    failsafe = [tint(trim_white(crop_figure(doc, FIGURES["failsafe"])), CARD)]
    print("  arch      %dx%d" % arch.size)
    print("  visual    " + " + ".join("%dx%d" % v.size for v in visual))
    print("  failsafe  " + " + ".join("%dx%d" % v.size for v in failsafe))

    sc = min((W - 144) / arch.width, (H - 370) / arch.height)
    if sc > 1.0:
        print("  NOTE: architecture figure upscaled %.2fx -- raise DPI" % sc)
    fw, fh = int(arch.width * sc), int(arch.height * sc)
    arch_s = arch.resize((fw, fh), Image.LANCZOS)
    fx = (W - fw) // 2
    fy = 196 + max(0, ((H - 330) - fh) // 2)

    report_pacing(args.arch_seconds)

    fps = args.fps
    xf = max(4, int(0.3 * fps))
    LA = int(args.arch_seconds * fps)
    LV = int(args.visual_seconds * fps)
    LF = int(args.failsafe_seconds * fps)

    segs = [
        (lambda i: figure_frame(arch_s, fx, fy, fw, fh, i / LA, args), LA),
        (lambda i: visual_frame(visual, i / LV, args, "Clinician review",
                                "Prediction, saliency and live re-run on one screen"), LV),
        (lambda i: visual_frame(failsafe, i / LF, args, "Fail-safe",
                                "A valid scan is classified; a non-MRI upload is refused"), LF),
    ]

    def stream():
        for si, (fn, L) in enumerate(segs):
            nfn, _ = segs[(si + 1) % len(segs)]
            for i in range(xf, L - xf):
                yield fn(i)
            for k in range(xf):
                t = ease_io((k + 1) / (xf + 1))
                yield Image.blend(fn(L - xf + k), nfn(k), t)

    count = sum(L - xf for _f, L in segs)
    poster_at = max(0, LA - 2 * xf - 2)
    return stream, count, poster_at


# ---------------------------------------------------------------- write
def write(stream, count, stem, fps, crf, poster_at):
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")
    mp4 = imageio.get_writer(
        stem + ".mp4", fps=fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", quality=None,
        output_params=["-crf", str(crf), "-preset", "slow", "-profile:v", "high",
                       "-movflags", "+faststart", "-an"])
    written = 0
    for i, f in enumerate(stream()):
        a = np.asarray(f)
        if a.shape != (H, W, 3) or a.dtype != np.uint8:
            sys.exit("frame %d is %s %s, expected (%d, %d, 3) uint8"
                     % (i, a.shape, a.dtype, H, W))
        mp4.append_data(a)
        written += 1
        if i == poster_at:
            f.save(stem + "_poster.jpg", quality=90, optimize=True, progressive=True)
        if i % 60 == 0:
            print("    frame %d/%d" % (i, count), end="\r", flush=True)
    mp4.close()
    print(" " * 34, end="\r")
    try:
        import imageio_ffmpeg
        got = sum(1 for fr in imageio_ffmpeg.read_frames(stem + ".mp4", bits_per_pixel=24)
                  if isinstance(fr, bytes))
        print("  %s: %d/%d frames decode" % (
            "VERIFIED" if got == written else "WARNING", got, written))
    except Exception as e:
        print("  verify skipped:", e)
    for f in (stem + ".mp4", stem + "_poster.jpg"):
        if os.path.exists(f):
            print("  %-30s %8.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--title", default="OneHealth+")
    ap.add_argument("--subtitle",
                    default="Explainable, clinician-in-the-loop brain tumor MRI classification")
    ap.add_argument("--final-caption",
                    default="Validate, classify, explain, then wait for the clinician")
    ap.add_argument("--arch-seconds", type=float, default=20.0)
    ap.add_argument("--visual-seconds", type=float, default=7.0)
    ap.add_argument("--failsafe-seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--dim", type=float, default=0.86)
    ap.add_argument("--grid", action="store_true",
                    help="write a coordinate overlay for Figure 3 and exit")
    ap.add_argument("--themes", action="store_true",
                    help="print the WCAG contrast table and exit")
    ap.add_argument("--out", default="onehealth")
    args = ap.parse_args()

    if args.themes:
        check_theme()
        return

    folder = os.path.dirname(os.path.abspath(args.pdf))
    if args.grid:
        doc = _fitz().open(args.pdf)
        grid_overlay(trim_white(crop_figure(doc, FIGURES["arch"])),
                     os.path.join(folder, "_grid.png"))
        return

    stream, count, poster_at = build(args)
    print("%d frames, %dx%d, %.1fs @ %d fps" % (count, W, H, count / args.fps, args.fps))
    write(stream, count, os.path.join(folder, args.out), args.fps, args.crf, poster_at)
    print("\ndone.")


if __name__ == "__main__":
    main()
