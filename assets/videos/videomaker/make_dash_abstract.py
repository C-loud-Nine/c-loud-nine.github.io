#!/usr/bin/env python3
r"""
make_dash_abstract.py

An animated graphical abstract for DASH: an annotated walk through the
architecture figure, then the teacher/student sample grids per dataset.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_dash_abstract.py "C:\Users\Shafi.09\Desktop\Portfolio\dash" \
        --figure "C:\Users\Shafi.09\Desktop\Portfolio\dash\dashdia.png"

    --grid      write a coordinate overlay for the figure and exit, so the
                STAGES boxes below can be read off instead of guessed
    --thumb     960x540 cut: cropped and magnified for a small thumbnail

Why side by side and not a wipe: teacher and student samples are drawn from
different seeds, so they are not pixel-aligned. Sweeping between them would
imply a correspondence that does not exist. Both panels stay on screen.
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- palette
# lifted from style.css so the clip sits on the page rather than on top of it
BG = (251, 246, 236)            # --bg          warm paper
INK = (52, 48, 42)              # --fg
MUTED = (138, 131, 119)         # --muted
ACCENT = (161, 61, 36)          # --accent      terracotta
PINE = (27, 58, 47)             # --heading     deep pine
SOFT = (230, 217, 196)          # --accent-soft
CARD = (255, 253, 250)

W, H = 1600, 900                # 16:9, matches .pub_thumb
PANEL_GAP = 128                 # the ratio badge is 112 across; 96 overlapped
SS = 2                          # overlay supersampling: crisp edges
FPS = 30
DPI = 220                       # PDF raster resolution

FONTS = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
FONTS_B = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_fc = {}


def F(size, bold=False):
    k = (size, bold)
    if k not in _fc:
        for p in (FONTS_B if bold else FONTS):
            if os.path.exists(p):
                try:
                    _fc[k] = ImageFont.truetype(p, size)
                    break
                except OSError:
                    pass
        else:
            _fc[k] = ImageFont.load_default()
    return _fc[k]


# ------------------------------------------- stages for the DASH figure
# (caption, [(x0, y0, x1, y1) as fractions of the figure])
#
# Read off dashdia.png (1117x502) with --grid. Boxes are sized to include
# each element's caption text, not just its outline: a highlight that cuts
# the words off under it looks like a misregistration.
STAGES = [
    ("Noise the clean image",
     [(0.080, 0.375, 0.250, 0.605)]),
    ("Teacher: two frozen passes",
     [(0.300, 0.270, 0.470, 0.930)]),
    ("TIRT weight transfer",
     [(0.408, 0.155, 0.562, 0.235)]),
    ("Two teacher scores",
     [(0.462, 0.375, 0.568, 0.705)]),
    ("Compact student, same structure",
     [(0.575, 0.285, 0.695, 0.915)]),
    ("Imitation and unconditional losses",
     [(0.634, 0.185, 0.715, 0.345), (0.675, 0.760, 0.755, 0.930)]),
    ("Anchor loss on true noise",
     [(0.765, 0.185, 0.908, 0.455)]),
    ("Guidance calibration, no CFG",
     [(0.772, 0.672, 0.908, 0.825)]),
]


def stage_schedule(stages):
    """Screen time per stage, proportional to its caption length.

    Uniform timing means a 3-word caption and a 5-word caption get the same
    hold, so the longest line always runs fastest. Weighting by word count
    spends the same total budget where reading actually needs it, and keeps
    the pacing correct if a caption is reworded later.
    """
    w = [max(3.0, len(c.split())) for c, _ in stages]
    tot = sum(w)
    out, acc = [], 0.0
    for x in w:
        out.append((acc / tot, (acc + x) / tot))
        acc += x
    return out


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


def pdf_to_image(path, dpi=DPI):
    pix = _fitz().open(path)[0].get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def load_figure(path):
    """Flatten onto the page colour, not white: dashdia.png is RGBA, and
    compositing it onto white leaves a bright rectangle sitting on cream."""
    if os.path.splitext(path)[1].lower() == ".pdf":
        return pdf_to_image(path, dpi=300)
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, BG)
        flat.paste(im, mask=im.split()[-1])
        return flat
    return im.convert("RGB")


def grid_overlay(fig, out):
    """Coordinate grid over the figure so STAGES boxes can be read off."""
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


# ---------------------------------------------------------------- datasets
# display name, teacher stem, student stem, metric chips
# Order matters: the first entry is the hook AND supplies the poster frame.
# ImageNet 64x64 leads because its 64px samples stay photographic at
# thumbnail scale, where CIFAR's 32px samples turn to mush, and it is the
# harder benchmark a reviewer weighs most.
DATASETS = [
    ("ImageNet 64x64", "in64_teacher_grid", "in64_student_grid",
     ["64x64 class-conditional", "single-branch student", "guidance calibrated"]),
    ("CIFAR-10", "cifar10_teacher_5x10", "cifar10_student_5x10",
     ["FID 8.87", "35.8M \u2192 6.1M", "no CFG at inference"]),
    ("CIFAR-100", "teacher_c100_diversity", "student_c100_diversity",
     ["FID 10.47", "5.9x fewer parameters", "~2x fewer forward passes"]),
]


# ---------------------------------------------------------------- helpers
def column_edges(im, n_expected=10):
    """Column boundaries from the grid's own white gutters, so a highlight
    lands on a class column instead of on an assumed even split."""
    a = np.asarray(im.convert("L"), float)
    body = a[:int(a.shape[0] * 0.88)]          # exclude the label strip
    gaps = body.mean(0) > 246
    runs, start = [], None
    for i, g in enumerate(gaps):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= 2:
                runs.append((start + i) / 2)
            start = None
    if start is not None and len(gaps) - start >= 2:
        runs.append((start + len(gaps)) / 2)
    if len(runs) != n_expected - 1:
        return None
    return [e / a.shape[1] for e in [0.0] + runs + [float(a.shape[1])]]


_shadow_cache = {}


def shadow(size, radius=14, blur=22, alpha=52):
    """Memoised: the panel geometry never changes, so blurring this once per
    frame was burning most of the render time."""
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


def chip(d, xy, text, size, fg, bg, alpha=1.0):
    f = F(size, True)
    pad = size * 0.55
    w = d.textlength(text, font=f) + 2 * pad
    h = size + 1.5 * pad
    x, y = xy
    d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=fade(bg, alpha),
                        outline=fade(SOFT, alpha), width=max(1, int(size * 0.05)))
    d.text((x + pad, y + pad * 0.72), text, font=f, fill=fade(fg, alpha))
    return w


# ---------------------------------------------------------------- frames
def figure_frame(fig_scaled, fx, fy, fw, fh, stages, t, args):
    """One frame of the figure walk. Earlier stages stay lit; the active one
    is spotlit and outlined."""
    canvas = Image.new("RGB", (W, H), BG)

    n = len(stages)
    intro, outro = 0.10, 0.12
    body = 1.0 - intro - outro
    sched = stage_schedule(stages)
    u = (t - intro) / body if body > 0 else 0.0
    k, local = 0, 0.0
    if t >= intro:
        for i, (a0, a1) in enumerate(sched):
            if u < a1 or i == n - 1:
                k, local = i, min(1.0, max(0.0, (u - a0) / (a1 - a0)))
                break
    final_hold = t > intro + body

    reveal = Image.new("L", (fw, fh), 0)
    rd = ImageDraw.Draw(reveal)
    if final_hold:
        reveal = Image.new("L", (fw, fh), 255)
    elif t >= intro:
        for i in range(k + 1):
            a = 255 if i < k else int(255 * ease_out(seg(local, 0.0, 0.45)))
            for (x0, y0, x1, y1) in stages[i][1]:
                rd.rounded_rectangle([x0 * fw, y0 * fh, x1 * fw, y1 * fh],
                                     radius=14, fill=a)
        reveal = reveal.filter(ImageFilter.GaussianBlur(26))

    dim = Image.new("RGB", (fw, fh), BG)
    shown = Image.composite(fig_scaled, Image.blend(fig_scaled, dim, args.dim), reveal)

    ia = ease_out(seg(t, 0.0, intro))
    if ia < 1.0:
        shown = Image.blend(Image.new("RGB", (fw, fh), BG), shown, ia)
    canvas.paste(shown, (fx, fy))

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS

    ti = ease_out(seg(t, 0.0, 0.08))
    d.text((72 * S, 36 * S), args.title, font=F(62 * S, True), fill=fade(PINE, ti))
    d.text((72 * S, 108 * S), args.subtitle, font=F(28 * S), fill=fade(MUTED, ti))
    d.rectangle([72 * S, 152 * S, (72 + ease_out(seg(t, 0.02, 0.12)) * 130) * S, 156 * S],
                fill=fade(ACCENT, ti))

    if not final_hold and t >= intro:
        oa = ease_out(seg(local, 0.0, 0.4)) * (1 - ease_out(seg(local, 0.90, 1.0)))
        for (x0, y0, x1, y1) in stages[k][1]:
            d.rounded_rectangle([(fx + x0 * fw) * S, (fy + y0 * fh) * S,
                                 (fx + x1 * fw) * S, (fy + y1 * fh) * S],
                                radius=10 * S, outline=fade(ACCENT, oa), width=4 * S)

    cap = args.final_caption if final_hold else stages[k][0]
    # hold the plate opaque across the beat: a caption fading out while you
    # are still reading it is worse than a hard change
    ca = 1.0 if final_hold else ease_out(seg(local, 0.04, 0.26))
    if ca > 0.01:
        f = F(36 * S, True)
        tw = d.textlength(cap, font=f) / S
        bx, by = (W - tw) / 2 - 34, H - 126
        d.rounded_rectangle([bx * S, by * S, (bx + tw + 68) * S, (by + 66) * S],
                            radius=33 * S, fill=fade(CARD, ca),
                            outline=fade(SOFT, ca), width=2 * S)
        d.text(((bx + 34) * S, (by + 16) * S), cap, font=f, fill=fade(PINE, ca))

    for i in range(n):
        r = 5 * S
        x = (W / 2 - (n - 1) * 11 + i * 22) * S
        y = (H - 48) * S
        on = (i <= k) and t >= intro
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=fade(ACCENT if (i == k and not final_hold) else
                            (PINE if on else MUTED), 1.0 if on else 0.28))

    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def compose(pair, args, t_local, t_global, idx, n):
    """One dataset frame: teacher and student grids side by side."""
    canvas = Image.new("RGB", (W, H), BG)
    tea, stu = pair["tea"], pair["stu"]
    pw, ph = tea.size
    gap = PANEL_GAP
    x0 = (W - (pw * 2 + gap)) // 2
    y0 = 252

    for k, (im, px) in enumerate(((tea, x0), (stu, x0 + pw + gap))):
        a = ease_out(seg(t_local, 0.06 + k * 0.10, 0.46 + k * 0.10))
        if a <= 0.01:
            continue
        dy = int((1 - a) * 26)
        card = round_corners(im, 12)
        sh, ox, oy = shadow(im.size)
        if a > 0.35:
            canvas.paste(sh, (px - ox, y0 + dy - oy), sh)
        if a < 1.0:
            base = canvas.crop((px, y0 + dy, px + pw, y0 + dy + ph)).convert("RGBA")
            card = Image.blend(base, card, a)
        canvas.paste(card, (px, y0 + dy), card)

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS

    ti = ease_out(seg(t_global, 0.0, 0.10)) if idx == 0 else 1.0
    dy = (1 - ti) * 16
    d.text((72 * S, (36 - dy) * S), args.title, font=F(62 * S, True), fill=fade(PINE, ti))
    d.text((72 * S, (108 - dy) * S), args.subtitle, font=F(28 * S), fill=fade(MUTED, ti))
    rw = (ease_out(seg(t_global, 0.03, 0.14)) if idx == 0 else 1.0) * 130
    d.rectangle([72 * S, 152 * S, (72 + rw) * S, 156 * S], fill=fade(ACCENT, ti))

    da = ease_out(seg(t_local, 0.02, 0.28))
    f = F(34 * S, True)
    tw = d.textlength(pair["name"], font=f) / S
    cw = tw + 58
    d.rounded_rectangle([(W - 72 - cw) * S, 46 * S, (W - 72) * S, 114 * S],
                        radius=34 * S, fill=fade(PINE, da))
    d.text(((W - 72 - cw + 29) * S, 62 * S), pair["name"], font=f,
           fill=fade((255, 255, 255), da))

    for k, (label, sub) in enumerate(((args.teacher_label, args.teacher_sub),
                                      (args.student_label, args.student_sub))):
        a = ease_out(seg(t_local, 0.14 + k * 0.10, 0.5 + k * 0.10))
        if a <= 0.01:
            continue
        px = x0 + k * (pw + gap)
        fl = F(36 * S, True)
        d.text((px * S, 196 * S), label, font=fl, fill=fade(PINE if k else INK, a))
        lw = d.textlength(label, font=fl) / S
        d.text(((px + lw + 14) * S, 208 * S), sub, font=F(25 * S), fill=fade(MUTED, a))

    ca = ease_out(seg(t_local, 0.42, 0.68))
    if ca > 0.01:
        cx, cy, r = x0 + pw + gap / 2, y0 + ph / 2, 56
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                  fill=fade(ACCENT, ca))
        f2 = F(34 * S, True)
        tw2 = d.textlength(args.ratio, font=f2) / S
        d.text(((cx - tw2 / 2) * S, (cy - 23) * S), args.ratio, font=f2,
               fill=fade((255, 255, 255), ca))
        for s_ in (-1, 1):
            d.line([((cx + s_ * (r + 7)) * S, cy * S),
                    ((cx + s_ * (r + 22 * ca + 7)) * S, cy * S)],
                   fill=fade(ACCENT, ca), width=4 * S)

    # class-column walk: columns are class-aligned in both grids even though
    # individual samples are not, so this comparison is honest
    cols = pair["cols"]
    if cols and t_local > 0.52:
        ncol = len(cols) - 1
        walk = (t_local - 0.52) / 0.48
        steps = 4
        k = min(steps - 1, int(walk * steps))
        sub = walk * steps - k
        glow = min(ease_out(sub / 0.25), ease_out((1 - sub) / 0.25), 1.0)
        ci = min(ncol - 1, int(k * ncol / steps) + 1)
        x_a, x_b = cols[ci], cols[ci + 1]
        for side in (0, 1):
            px = x0 + side * (pw + gap)
            gx0, gx1 = px + x_a * pw, px + x_b * pw
            for a_, b_ in ((px, gx0), (gx1, px + pw)):
                d.rectangle([a_ * S, y0 * S, b_ * S, (y0 + ph * 0.88) * S],
                            fill=BG + (int(96 * glow),))
            d.rounded_rectangle([gx0 * S, y0 * S, gx1 * S, (y0 + ph * 0.88) * S],
                                radius=5 * S, outline=fade(ACCENT, glow), width=4 * S)

    cy_ = y0 + ph + 52
    cx_ = x0
    for i, text in enumerate(pair["chips"]):
        a = ease_out(seg(t_local, 0.55 + i * 0.09, 0.85 + i * 0.09))
        if a <= 0.01:
            continue
        cx_ += chip(d, (cx_ * S, (cy_ + (1 - a) * 8) * S), text, 30 * S,
                    PINE, CARD, a) / S + 18

    for i in range(n):
        r = 5 * S
        x = (W - 72 - (n - 1 - i) * 22) * S
        y = (H - 62) * S
        on = i == idx
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=fade(ACCENT if on else MUTED, 1.0 if on else 0.32))

    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def thumb_frame(pair, args, t_local):
    """A 180px-wide thumbnail cannot show a 5x10 grid: it becomes texture.
    Crop to the first few class columns and magnify, so the shape still
    reads as 'two sets of samples' at postage-stamp size. Crops come from
    the full-resolution render, never from the downscaled panel."""
    TW, TH = 960, 540
    canvas = Image.new("RGB", (TW, TH), BG)

    cols = pair["cols"]
    n = args.thumb_classes
    x1 = cols[n] if (cols and len(cols) > n) else n / 10.0
    rows = args.thumb_rows / 5.0

    gap = 112                   # the ratio badge is 84px across; 28 overlapped both panels
    pw = (TW - 40 - gap) // 2
    crops = []
    for im in (pair["tea_full"], pair["stu_full"]):
        c = im.crop((0, 0, int(im.width * x1), int(im.height * 0.88 * rows)))
        crops.append(c.resize((pw, round(c.height * pw / c.width)), Image.LANCZOS))
    ph = crops[0].height
    y0 = (TH - ph) // 2 + 22

    for k, c in enumerate(crops):
        a = ease_out(seg(t_local, 0.04 + k * 0.08, 0.40 + k * 0.08))
        if a <= 0.01:
            continue
        px = 20 + k * (pw + gap)
        card = round_corners(c, 8)
        if a < 1.0:
            base = canvas.crop((px, y0, px + pw, y0 + ph)).convert("RGBA")
            card = Image.blend(base, card, a)
        canvas.paste(card, (px, y0), card)

    ov = Image.new("RGBA", (TW * SS, TH * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    for k, lab in enumerate((args.thumb_teacher, args.thumb_student)):
        a = ease_out(seg(t_local, 0.08 + k * 0.08, 0.44 + k * 0.08))
        d.text(((20 + k * (pw + gap)) * S, (y0 - 54) * S), lab, font=F(36 * S, True),
               fill=fade(PINE if k else INK, a))
    ba = ease_out(seg(t_local, 0.40, 0.62))
    if ba > 0.01:
        cx, cy, r = TW / 2, y0 + ph / 2, 42
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                  fill=fade(ACCENT, ba))
        f = F(28 * S, True)
        tw = d.textlength(args.ratio, font=f) / S
        d.text(((cx - tw / 2) * S, (cy - 19) * S), args.ratio, font=f,
               fill=fade((255, 255, 255), ba))
    na = ease_out(seg(t_local, 0.50, 0.70))
    f2 = F(28 * S, True)
    tw2 = d.textlength(pair["name"], font=f2) / S
    d.text((((TW - tw2) / 2) * S, (TH - 56) * S), pair["name"], font=f2,
           fill=fade(MUTED, na))

    ov = ov.resize((TW, TH), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


# ---------------------------------------------------------------- build
def build(args):
    avail = []
    for name, tstem, sstem, chips in DATASETS:
        tp = os.path.join(args.folder, tstem + ".pdf")
        sp = os.path.join(args.folder, sstem + ".pdf")
        if os.path.exists(tp) and os.path.exists(sp):
            avail.append((name, tp, sp, chips))
        else:
            miss = tp if not os.path.exists(tp) else sp
            print("  skipping %s: missing %s" % (name, os.path.basename(miss)))
    if not avail:
        sys.exit("no teacher/student PDF pairs found in " + args.folder)
    print("%d dataset(s): %s" % (len(avail), ", ".join(a[0] for a in avail)))

    panel_w = (W - 88 - PANEL_GAP) // 2
    rendered = []
    for name, tp, sp, chips in avail:
        tea_full, stu_full = pdf_to_image(tp), pdf_to_image(sp)
        h = round(tea_full.height * panel_w / tea_full.width)
        tea = tea_full.resize((panel_w, h), Image.LANCZOS)
        stu = stu_full.resize((panel_w, h), Image.LANCZOS)
        cols = column_edges(tea_full)          # measured at full resolution
        if cols is None:
            print("  %s: column gutters not detected, highlight disabled" % name)
        rendered.append(dict(name=name, tea=tea, stu=stu, chips=chips, cols=cols,
                             tea_full=tea_full, stu_full=stu_full))

    n = len(rendered)
    total = int(args.seconds * args.fps)
    per = max(int(2.6 * args.fps), total // n)
    xf = max(4, int(0.3 * args.fps))

    if args.thumb:
        def tstream():
            for idx in range(n):
                for i in range(per - xf):
                    yield thumb_frame(rendered[idx], args, i / per)
                nxt = rendered[(idx + 1) % n]
                for k in range(xf):
                    t = ease_io((k + 1) / (xf + 1))
                    yield Image.blend(
                        thumb_frame(rendered[idx], args, (per - xf + k) / per),
                        thumb_frame(nxt, args, k / per), t)
        return tstream, per * n, (960, 540), per * n // 2

    fig_geom = None
    if args.figure:
        fig = load_figure(args.figure)
        sc = min((W - 144) / fig.width, (H - 320) / fig.height)
        fw, fh = int(fig.width * sc), int(fig.height * sc)
        fig_geom = (fig.resize((fw, fh), Image.LANCZOS), (W - fw) // 2, 180, fw, fh)
        print("  figure: %s %dx%d -> %dx%d, %d stages"
              % (os.path.basename(args.figure), fig.width, fig.height, fw, fh, len(STAGES)))

    def opening(t):
        g, fx, fy, fw, fh = fig_geom
        return figure_frame(g, fx, fy, fw, fh, STAGES, t, args)

    m_frames = 0 if (args.no_figure or not fig_geom) else int(args.figure_seconds * args.fps)

    def at(idx, i):
        return compose(rendered[idx], args, i / per, (idx * per + i) / (per * n), idx, n)

    # Segments in play order, then one generic cross-fade rule. Each segment
    # emits [xf, L-xf) as body and [L-xf, L) blended into the next segment's
    # [0, xf) -- so no segment's opening frames are ever shown twice, which is
    # what makes a hand-rolled cross-fade stutter.
    segs = [(lambda i, j=idx: at(j, i), per) for idx in range(n)]
    if m_frames:
        segs.append((lambda i: opening(i / m_frames), m_frames))

    def stream():
        for si, (fn, L) in enumerate(segs):
            nfn, _ = segs[(si + 1) % len(segs)]
            for i in range(xf, L - xf):
                yield fn(i)
            for k in range(xf):
                t = ease_io((k + 1) / (xf + 1))
                yield Image.blend(fn(L - xf + k), nfn(k), t)

    # poster: the end of the first dataset beat, just before its cross-fade.
    # The middle of the beat catches the ratio badge still fading in and no
    # metric chips at all; a fixed fraction of the whole clip lands inside a
    # cross-fade and posters as a double exposure.
    poster_at = max(0, per - 2 * xf - 2)
    count = sum(L - xf for _fn, L in segs)
    return stream, count, (W, H), poster_at


# ---------------------------------------------------------------- write
def write(stream, count, stem, fps, crf, poster_at):
    """Stream frames straight into the encoder; memory stays flat."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")

    # H.264 only: the one codec every browser plays, and +faststart lets it
    # start before it has finished downloading.
    mp4 = imageio.get_writer(
        stem + ".mp4", fps=fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", quality=None,
        output_params=["-crf", str(crf), "-preset", "slow", "-profile:v", "high",
                       "-movflags", "+faststart", "-an"])
    written = 0
    for i, f in enumerate(stream()):
        a = np.asarray(f)
        # A frame of the wrong size or mode does not make the encoder fail
        # loudly: it writes a container with a correct duration and garbage
        # NAL lengths, which only shows up when something tries to play it.
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

    # Decode the file back and count frames. A short or corrupt encode still
    # produces a valid-looking container reporting the right duration, so
    # "the file exists and ffprobe reads it" is not evidence that it plays.
    try:
        import imageio_ffmpeg
        got = sum(1 for fr in imageio_ffmpeg.read_frames(stem + ".mp4",
                                                         bits_per_pixel=24)
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
    ap.add_argument("folder")
    ap.add_argument("--title", default="DASH")
    ap.add_argument("--subtitle", default="Dual-branch score distillation for "
                                          "guidance-calibrated compact diffusion models")
    ap.add_argument("--teacher-label", default="Teacher")
    ap.add_argument("--teacher-sub", default="35.8M parameters, CFG at inference")
    ap.add_argument("--student-label", default="DASH student")
    ap.add_argument("--student-sub", default="6.1M parameters, no CFG")
    ap.add_argument("--ratio", default="5.9x")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--figure", help="architecture figure (png/jpg/pdf) animated "
                                     "as the opening beat, e.g. dashdia.png")
    ap.add_argument("--figure-seconds", type=float, default=14.0)
    ap.add_argument("--no-figure", action="store_true", help="skip the figure walk")
    ap.add_argument("--final-caption",
                    default="Guidance distilled, not applied at inference")
    ap.add_argument("--dim", type=float, default=0.86,
                    help="how far unrevealed parts of the figure fade back, 0-1")
    ap.add_argument("--grid", action="store_true",
                    help="write a coordinate overlay for the figure and exit")
    ap.add_argument("--thumb", action="store_true",
                    help="960x540 cut: cropped, magnified, big labels")
    ap.add_argument("--thumb-classes", type=int, default=4)
    ap.add_argument("--thumb-rows", type=int, default=3)
    ap.add_argument("--thumb-teacher", default="Teacher 35.8M")
    ap.add_argument("--thumb-student", default="Student 6.1M")
    ap.add_argument("--out", default="dash")
    args = ap.parse_args()

    if args.grid:
        if not args.figure:
            sys.exit("--grid needs --figure")
        grid_overlay(load_figure(args.figure),
                     os.path.join(args.folder, "_grid.png"))
        return

    stream, count, size, poster_at = build(args)
    print("%d frames, %dx%d, %.1fs @ %d fps"
          % (count, size[0], size[1], count / args.fps, args.fps))
    write(stream, count, os.path.join(args.folder, args.out), args.fps, args.crf, poster_at)
    print("\ndone.")


if __name__ == "__main__":
    main()
