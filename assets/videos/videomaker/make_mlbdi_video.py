#!/usr/bin/env python3
r"""
make_mlbdi_video.py

An animated graphical abstract for the multi-level decoder interaction
paper (breast ultrasound): the qualitative strip, the comparison table
typeset from data, then the architecture annotated stage by stage.

    pip install pymupdf pillow numpy imageio imageio-ffmpeg
    python make_mlbdi_video.py "papers/mlbdi.pdf"

    --grid     coordinate overlay for Figure 1, so the STAGES boxes
               below can be read off instead of guessed
    --themes   WCAG contrast table
    --verify   check every TABLES value against the PDF text

Deliberately tight: three beats, about twenty-three seconds.

Table 1 reports IoU/Dice/Acc/F1 for BUSI and BUSI-WHU; the columns
taken here are BUSI IoU, BUSI Acc and BUSI-WHU IoU. The transformer
baselines report no classification accuracy, so their Acc cell is the
paper's own en-dash rather than a blank that would read as an omission.
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- palette
BG = (241, 243, 240)            # pale mineral, distinct from the siblings
INK = (26, 32, 30)
MUTED = (100, 110, 106)
ACCENT = (26, 104, 96)          # deep teal-green
PINE = (20, 62, 58)
SOFT = (214, 222, 218)
CARD = (253, 254, 253)

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
    """WCAG ratios against the page ground. Body text wants >= 4.5, large
    headings >= 3.0."""
    ok = True
    for name, c, need in (("ink", INK, 4.5), ("heading", PINE, 3.0),
                          ("muted", MUTED, 4.5), ("accent", ACCENT, 3.0)):
        r = contrast(c, BG)
        ok &= r >= need
        print("  %-9s %5.2f  %s" % (name, r, "ok" if r >= need else "FAIL (needs %.1f)" % need))
    print("all pass" if ok else "SOME FAIL")
    return ok


# ---------------------------------------- stages for the BTI-Net figure
# Read off Figure 1 with --grid. Boxes include each module's caption text,
# not just its outline: a highlight that cuts the words off under it looks
# like a misregistration. Each stage is a LIST of boxes so a module split
# across the figure can be lit together.
STAGES = [
    ("A 224x224 breast ultrasound frame",
     [(0.000, 0.180, 0.110, 0.760)]),
    ("Encoder stages with multi-scale fusion",
     [(0.110, 0.030, 0.360, 0.960)]),
    ("TIM exchanges features both ways",
     [(0.405, 0.130, 0.560, 0.700)]),
    ("UPA weights the exchange per sample",
     [(0.562, 0.130, 0.690, 0.700)]),
    ("One pass, mask and label",
     [(0.700, 0.000, 1.000, 0.990)]),
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


# ---------------------------------------------------------------- figures
# (page index, x0, y0, x1, y1) as fractions of the page
FIGURES = {
    # y1 runs to 0.300: at 0.285 the classification-label circles along the
    # bottom of Figure 1 were sliced in half
    "arch":   (2, 0.225, 0.150, 0.800, 0.300),
    "visual": (6, 0.250, 0.135, 0.782, 0.310),
}

# ---------------------------------------------------------------- tables
#   cols  (label, align, bar)   bar=True draws a proportional value bar
#   ours  index of the row to highlight, revealed last
#
# Table 5 reports IoU and Acc for each of the three datasets; the values
# below are its three IoU columns. Verified row by row against the text
# layer -- see --verify.
TABLES = {
    "joint": dict(
        cols=[("Method", "l", False), ("BUSI IoU", "r", True),
              ("BUSI Acc", "r", True), ("WHU IoU", "r", True)],
        rows=[["U-Net", 66.80, 86.50, 77.50],
              ["Attention U-Net", 68.20, 87.80, 79.20],
              ["UNet++", 69.50, 88.21, 80.70],
              ["MISSFormer", 72.80, "-", 84.60],
              ["MTAN", 68.90, 87.20, 80.00],
              ["MTANet", 72.10, 89.30, 83.70],
              ["MTL-OCA", 72.90, 89.90, 84.50],
              ["Proposed", 74.50, 90.60, 86.40]],
        ours=7, dec=2,
        note="Ahead on both datasets, against CNN, transformer and MTL baselines"),
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


def mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


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


def verify_tables(doc):
    """Every printed value must appear in the paper's text layer.

    Blind spot worth knowing: this cannot see numbers rasterised inside a
    figure, so it checks TABLES only, never the captions.
    """
    import re
    flat = re.sub(r"\s+", " ", " ".join(p.get_text() for p in doc))
    bad = []
    for key, spec in TABLES.items():
        for row in spec["rows"]:
            for v in row[1:]:
                if isinstance(v, str):
                    continue
                if ("%.*f" % (spec["dec"], v)) not in flat:
                    bad.append("%s / %s / %s" % (key, row[0], v))
    if bad:
        print("VALUES NOT FOUND IN THE PDF:")
        for b in bad:
            print("   ", b)
        return False
    n = sum(len(s["rows"]) * (len(s["cols"]) - 1) for s in TABLES.values())
    print("  %d table values all present in the paper text" % n)
    return True


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
    ft, _ = fit(d, args.title, 62, True, W - 520, S)
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
    """The architecture walk. Earlier stages stay lit; the active one is
    spotlit and outlined."""
    canvas = Image.new("RGB", (W, H), BG)
    n = len(STAGES)
    intro, outro = 0.04, 0.12
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
    ti = ease_out(seg(t, 0.0, 0.06))
    header(d, S, args, ti)
    d.rectangle([72 * S, 152 * S, (72 + ease_out(seg(t, 0.02, 0.10)) * 130) * S, 156 * S],
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


def table_frame(spec, head, t, args):
    """A typeset table: rows arrive staggered, the proposed row last, with a
    value bar per numeric cell scaled across that column's own range."""
    cols, rows, ours = spec["cols"], spec["rows"], spec["ours"]
    canvas = Image.new("RGB", (W, H), BG)
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    S = SS
    header(d, S, args, 1.0)
    d.rectangle([72 * S, 152 * S, (72 + 130) * S, 156 * S], fill=fade(ACCENT, 1))
    badge(d, S, head, ease_out(seg(t, 0.02, 0.26)))

    fh_, fb = F(34 * S), F(34 * S, True)
    label_w = max(d.textlength(str(r[0]), font=fb) for r in rows) / S + 76
    num_w = 226
    table_w = label_w + num_w * (len(cols) - 1)
    x0 = (W - table_w) / 2
    row_h = 66
    body_h = row_h * (len(rows) + 1) + 24
    # centre the card in the band between the header rule and the caption
    # plate, so a four-row table does not sit high with a hole beneath it
    band_top, band_bottom = 196, H - 152
    top = band_top + max(0, (band_bottom - band_top - body_h) / 2)

    card_a = ease_out(seg(t, 0.03, 0.30))
    if card_a > 0.02:
        pad = 38
        plate = Image.new("RGB", (int(table_w + 2 * pad), int(body_h + 2 * pad)),
                          mix(BG, (255, 255, 255), 0.6))
        sh, ox, oy = shadow(plate.size)
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
        tx = cx + 16 if align == "l" else cx + cw - tw - 16
        d.text((tx * S, hy * S), name, font=fb, fill=fade(MUTED, card_a))
    d.rectangle([x0 * S, (hy + 50) * S, (x0 + table_w) * S, (hy + 52) * S],
                fill=fade(MUTED, 0.45 * card_a))

    spans = {}
    for i in range(1, len(cols)):
        vals = [r[i] for r in rows if not isinstance(r[i], str)]
        if vals:
            spans[i] = (min(vals), max(vals))

    order = [i for i in range(len(rows)) if i != ours] + [ours]
    # build finishes by mid-beat: a card still filling in at t=0.76 reads as
    # broken at thumbnail size, where the type is illegible anyway
    step = min(0.045, 0.30 / max(1, len(rows)))
    for slot, ri in enumerate(order):
        a = ease_out(seg(t, 0.08 + slot * step, 0.26 + slot * step))
        if a <= 0.02:
            continue
        is_ours = ri == ours
        y = hy + 64 + ri * row_h
        dx = (1 - a) * 26
        if is_ours:
            d.rounded_rectangle([(x0 - 14) * S, (y - 10) * S,
                                 (x0 + table_w + 14) * S, (y + row_h - 16) * S],
                                radius=12 * S, fill=fade(ACCENT, 0.10 * a))
        for i, (_, align, bar) in enumerate(cols):
            cx = x0 + (0 if i == 0 else label_w + num_w * (i - 1)) - dx
            cw = label_w if i == 0 else num_w
            val = rows[ri][i]
            txt = str(val) if (i == 0 or isinstance(val, str)) else "%.*f" % (spec["dec"], val)
            f = fb if is_ours else fh_
            if i and bar and i in spans and not isinstance(val, str):
                lo, hi = spans[i]
                frac = 0.12 + 0.88 * (val - lo) / max(1e-6, hi - lo)
                bw = (cw - 32) * frac * ease_out(seg(t, 0.14 + slot * step,
                                                     0.40 + slot * step))
                d.rounded_rectangle([(cx + cw - 16 - bw) * S, (y + 46) * S,
                                     (cx + cw - 16) * S, (y + 55) * S], radius=5 * S,
                                    fill=fade(ACCENT if is_ours else MUTED,
                                              (0.85 if is_ours else 0.38) * a))
            tw = d.textlength(txt, font=f) / S
            tx = cx + 16 if align == "l" else cx + cw - tw - 16
            d.text((tx * S, y * S), txt, font=f, fill=fade(PINE if is_ours else INK, a))

    caption_plate(d, S, spec["note"], ease_out(seg(t, 0.54, 0.72)))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


def split_grid(fig, header_frac=0.055):
    """Cut a tall qualitative grid into two halves, each keeping the column
    header row. Fitted whole into a 16:9 band it renders at a third of the
    frame; side by side the cells are roughly twice the size."""
    w, h = fig.size
    hh = int(h * header_frac)
    head = fig.crop((0, 0, w, hh))
    body_h = h - hh
    half = hh + body_h // 2
    top = fig.crop((0, 0, w, half))
    bot = Image.new("RGB", (w, hh + (h - half)), CARD)
    bot.paste(head, (0, 0))
    bot.paste(fig.crop((0, half, w, h)), (0, hh))
    return top, bot


def visual_frame(halves, t, args):
    """The qualitative grid, in two halves side by side."""
    canvas = Image.new("RGB", (W, H), BG)
    # 26, not 16: a column header that runs to the edge of its own crop
    # ends up 16px from the card edge and reads as clipped even though the
    # glyph is whole
    gap, pad = 40, 26
    n_panels = max(1, len(halves))
    avail_w = (W - 150 - gap * (n_panels - 1)) // n_panels
    avail_h = H - 360          # clears the caption plate at H-126
    z = 1.0 + 0.03 * ease_io(t)
    scaled = []
    for hf in halves:
        s_ = min(avail_w / hf.width, avail_h / hf.height)
        fw, fh = int(hf.width * s_), int(hf.height * s_)
        zw, zh = int(fw * z), int(fh * z)
        im = hf.resize((zw, zh), Image.LANCZOS).crop(
            ((zw - fw) // 2, (zh - fh) // 2, (zw - fw) // 2 + fw, (zh - fh) // 2 + fh))
        scaled.append(im)
    tw_ = sum(i.width + 2 * pad for i in scaled) + gap * (len(scaled) - 1)
    x = (W - tw_) // 2
    fy = 190

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
    badge(d, S, "BUSI", ease_out(seg(t, 0.02, 0.26)))
    caption_plate(d, S, args.visual_caption, ease_out(seg(t, 0.30, 0.52)))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")


# ---------------------------------------------------------------- build
def build(args):
    doc = _fitz().open(args.pdf)
    verify_tables(doc)
    arch = tint(trim_white(crop_figure(doc, FIGURES["arch"])), BG)
    vis = tint(trim_white(crop_figure(doc, FIGURES["visual"])), CARD)
    # split only a tall grid. A landscape figure already fills a 16:9 band,
    # and halving one cuts a row and repeats its header.
    visual = split_grid(vis) if vis.height > vis.width * 0.85 else [vis]
    print("  arch    %dx%d" % arch.size)
    print("  visual  " + " + ".join("%dx%d" % v.size for v in visual))

    sc = min((W - 144) / arch.width, (H - 330) / arch.height)
    fw, fh = int(arch.width * sc), int(arch.height * sc)
    arch_s = arch.resize((fw, fh), Image.LANCZOS)
    fx = (W - fw) // 2
    fy = 196 + max(0, ((H - 330) - fh) // 2)

    fps = args.fps
    xf = max(4, int(0.3 * fps))
    LA = int(args.arch_seconds * fps)
    LT = int(args.table_seconds * fps)
    LV = int(args.visual_seconds * fps)

    # the qualitative grid leads: at 376px it is the only beat that reads,
    # and it supplies the poster
    segs = [
        (lambda i: visual_frame(visual, i / LV, args), LV),
        (lambda i: table_frame(TABLES["joint"], "Versus baselines", i / LT, args), LT),
        (lambda i: figure_frame(arch_s, fx, fy, fw, fh, i / LA, args), LA),
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
    poster_at = max(0, LV - 2 * xf - 2)
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
    ap.add_argument("--title", default="Multi-Level Decoder Interaction")
    ap.add_argument("--subtitle",
                    default="Uncertainty-aware task coordination for breast ultrasound")
    ap.add_argument("--final-caption",
                    default="Tasks that refine each other as the mask is rebuilt")
    ap.add_argument("--visual-caption",
                    default="Sharper boundaries under posterior shadowing")
    ap.add_argument("--arch-seconds", type=float, default=12.0)
    ap.add_argument("--table-seconds", type=float, default=6.5)
    ap.add_argument("--visual-seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--dim", type=float, default=0.80)
    ap.add_argument("--grid", action="store_true",
                    help="write a coordinate overlay for Figure 1 and exit")
    ap.add_argument("--themes", action="store_true",
                    help="print the WCAG contrast table and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check every TABLES value against the PDF text and exit")
    ap.add_argument("--out", default="mlbdi")
    args = ap.parse_args()

    if args.themes:
        sys.exit(0 if check_theme() else 1)

    folder = os.path.dirname(os.path.abspath(args.pdf))
    if args.grid:
        doc = _fitz().open(args.pdf)
        grid_overlay(trim_white(crop_figure(doc, FIGURES["arch"])),
                     os.path.join(folder, "_grid.png"))
        return
    if args.verify:
        sys.exit(0 if verify_tables(_fitz().open(args.pdf)) else 1)

    stream, count, poster_at = build(args)
    print("%d frames, %dx%d, %.1fs @ %d fps" % (count, W, H, count / args.fps, args.fps))
    write(stream, count, os.path.join(folder, args.out), args.fps, args.crf, poster_at)
    print("\ndone.")


if __name__ == "__main__":
    main()
