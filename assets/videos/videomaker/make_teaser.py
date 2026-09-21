#!/usr/bin/env python3
r"""
make_teaser.py

Animated graphical abstract from Blur_/Sharp_ pairs, on the site's own warm
paper palette. Writes <out>.mp4, <out>_poster.jpg and <out>_abstract.png next
to the images.

    pip install pillow numpy imageio imageio-ffmpeg
    python make_teaser.py "C:\Users\Shafi.09\Desktop\Portfolio\iphone"

Options:
    --seconds 11          total loop length
    --order 2190,2983     which pairs, in which order
    --tiers Hard,Medium   REAL tier per pair, in --order order. Omit it
                          rather than guess: the chip is read as a claim.
    --dark                render on the site's dark-mode palette instead
    --title / --subtitle / --footer / --footer-right
    --no-abstract         skip the still figure

Design notes

  * palette is lifted from style.css (--bg, --heading, --accent, --muted) so
    the video sits on the page instead of on top of it.
  * every vector and glyph is drawn on a 2x layer and LANCZOS-downsampled;
    photographs stay at native resolution and are never supersampled.
  * the magnified crop is chosen where blur ACTUALLY differs from sharp.
    Ranking by texture energy in the sharp frame -- the obvious choice --
    picks foliage every time, which is high-frequency in BOTH frames and so
    demonstrates nothing.
  * crops are cut from the original full-resolution file, not from the
    downscaled stage copy.
  * PSNR / SSIM / edge-energy loss are measured per pair. Nothing on the
    canvas is a hard-coded number.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

SS = 2                          # supersample factor for the overlay layers

# ---------------------------------------------------------------- palettes
LIGHT = dict(                   # style.css :root
    bg=(251, 246, 236),         # --bg          warm paper
    ink=(52, 48, 42),           # --fg
    head=(27, 58, 47),          # --heading     deep pine
    accent=(161, 61, 36),       # --accent      terracotta
    muted=(138, 131, 119),      # --muted
    soft=(230, 217, 196),       # --accent-soft
    shadow=58,
)
DARK = dict(                    # style.css prefers-color-scheme: dark
    bg=(23, 22, 20),
    ink=(204, 199, 189),
    head=(159, 212, 187),
    accent=(216, 121, 93),
    muted=(125, 120, 111),
    soft=(58, 53, 45),
    shadow=150,
)
P = LIGHT                       # rebound in main()

# ----------------------------------------------------------------- canvas
CW, CH     = 1920, 1080
M          = 64                 # page margin
HEADER_H   = 150
STAGE_X    = M
STAGE_Y    = 190
STAGE_W    = 1198
STAGE_H    = 674
PANEL_X    = 1310
PANEL_W    = 546
PANEL_H    = 307
PANEL_GAP  = 60                 # room for the lower panel's caption
PANEL_Y1   = STAGE_Y
PANEL_Y2   = STAGE_Y + PANEL_H + PANEL_GAP
CROP_FRAC  = 0.22
FPS        = 30

# .pub_thumb renders at 180x100, so the thumbnail loop is a 2x asset. None of
# the full canvas survives at that size -- 46px metric numbers land at 4px --
# so the thumbnail is the wipe alone, framed tight on the moving subject.
THUMB_W, THUMB_H = 360, 200
THUMB_ZOOM = 2.2

LEFT_LABEL  = "Blurred input"
RIGHT_LABEL = "Sharp reference"

FONTS = {
    "bold":  [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "reg":   [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "light": [r"C:\Windows\Fonts\segoeuisl.ttf", r"C:\Windows\Fonts\segoeui.ttf",
              r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono":  [r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\cour.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
}
_fc = {}


def F(size, weight="reg"):
    key = (size, weight)
    if key not in _fc:
        for p in FONTS[weight]:
            if os.path.exists(p):
                try:
                    _fc[key] = ImageFont.truetype(p, size)
                    break
                except OSError:
                    pass
        else:
            _fc[key] = ImageFont.load_default()
    return _fc[key]


def rgba(c, a=1.0):
    return tuple(c) + (int(max(0, min(255, round(a * 255)))),)


# -------------------------------------------------------------------- pen
class Pen:
    """An RGBA overlay drawn at SSx and downsampled on .flatten().

    Callers work entirely in 1x page coordinates; the scale factor lives
    here. Drawing vectors at 1x is what makes a figure look cheap.
    """

    def __init__(self, w, h, ss=SS):
        self.ss = ss
        self.img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def _f(self, sz, w):
        return F(int(round(sz * self.ss)), w)

    def width(self, s, sz, w="reg", sp=0.0):
        f = self._f(sz, w)
        if not s:
            return 0.0
        return (sum(self.d.textlength(c, font=f) + sp * self.ss for c in s)
                - sp * self.ss) / self.ss

    def text(self, xy, s, sz, w, fill, right=False, sp=0.0):
        S, f = self.ss, self._f(sz, w)
        x = xy[0] * S
        if right:
            x -= self.width(s, sz, w, sp) * S
        y = xy[1] * S
        if sp:
            for c in s:
                self.d.text((x, y), c, font=f, fill=fill)
                x += self.d.textlength(c, font=f) + sp * S
        else:
            self.d.text((x, y), s, font=f, fill=fill)

    def line(self, pts, fill, w=1.0):
        S = self.ss
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill,
                    width=max(1, int(round(w * S))))

    def rect(self, box, fill=None, outline=None, w=1.0, r=0):
        S = self.ss
        b = [box[0] * S, box[1] * S, box[2] * S, box[3] * S]
        if r:
            self.d.rounded_rectangle(b, radius=r * S, fill=fill,
                                     outline=outline, width=max(1, int(round(w * S))))
        else:
            self.d.rectangle(b, fill=fill, outline=outline,
                             width=max(1, int(round(w * S))))

    def ellipse(self, box, fill=None, outline=None, w=1.0):
        S = self.ss
        self.d.ellipse([box[0] * S, box[1] * S, box[2] * S, box[3] * S],
                       fill=fill, outline=outline, width=max(1, int(round(w * S))))

    def polygon(self, pts, fill):
        S = self.ss
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def flatten(self):
        return self.img.resize((self.img.width // self.ss, self.img.height // self.ss),
                               Image.LANCZOS)


def over(base_rgb, ov_rgba, at=(0, 0)):
    """Composite an RGBA overlay onto part of an RGB image."""
    if at == (0, 0) and ov_rgba.size == base_rgb.size:
        return Image.alpha_composite(base_rgb.convert("RGBA"), ov_rgba).convert("RGB")
    box = (at[0], at[1], at[0] + ov_rgba.width, at[1] + ov_rgba.height)
    reg = Image.alpha_composite(base_rgb.crop(box).convert("RGBA"), ov_rgba)
    out = base_rgb.copy()
    out.paste(reg.convert("RGB"), at)
    return out


def soft_shadow(canvas, box, blur=20, spread=7, dy=6, r=4):
    """Shadow under a photo panel. Built once per pair, never per frame."""
    lay = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(lay).rounded_rectangle(
        [box[0] - spread, box[1] - spread + dy,
         box[2] + spread, box[3] + spread + dy],
        radius=r + spread, fill=P["shadow"])
    canvas.paste(Image.new("RGB", canvas.size, (0, 0, 0)), (0, 0),
                 lay.filter(ImageFilter.GaussianBlur(blur)))


# ---------------------------------------------------------------- metrics
def box_filter(a, r):
    H, W = a.shape
    pad = np.pad(a, ((r, r), (r, r)), mode="reflect")
    ii = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return (ii[n:n + H, n:n + W] - ii[0:H, n:n + W]
            - ii[n:n + H, 0:W] + ii[0:H, 0:W]) / (n * n)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a, b, r=5):
    """Box-window SSIM on luma: within ~0.005 of the gaussian-window value
    and needs no scipy."""
    x = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
    y = np.asarray(Image.fromarray(b).convert("L"), dtype=np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = box_filter(x, r), box_filter(y, r)
    sxx = box_filter(x * x, r) - mx * mx
    syy = box_filter(y * y, r) - my * my
    sxy = box_filter(x * y, r) - mx * my
    return float(np.mean(((2 * mx * my + C1) * (2 * sxy + C2)) /
                         ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))))


def grad_energy(im):
    g = np.asarray(im.convert("L"), dtype=np.float64)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def hf_loss(blur, sharp):
    eb, es = grad_energy(blur).mean(), grad_energy(sharp).mean()
    return max(0.0, 1.0 - eb / es) if es > 0 else 0.0


# ----------------------------------------------------------------- images
def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def cover(im, W, H):
    """Fill WxH exactly, centre-cropping the overflow."""
    s = max(W / im.width, H / im.height)
    r = im.resize((max(W, int(round(im.width * s))),
                   max(H, int(round(im.height * s)))), Image.LANCZOS)
    return r.crop(((r.width - W) // 2, (r.height - H) // 2,
                   (r.width - W) // 2 + W, (r.height - H) // 2 + H))


def _sat(a):
    """Summed-area table, zero-padded so window sums need no bounds checks."""
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _win_mean(ii, th, tw):
    """Mean over every th x tw window, one array op per corner."""
    return (ii[th:, tw:] - ii[:-th, tw:]
            - ii[th:, :-tw] + ii[:-th, :-tw]) / (th * tw)


def best_crop(blur, sharp, frac=CROP_FRAC):
    """Centre of the tile where deblurring is most visible: ranked by mean
    |blur - sharp|, damped by sharp-side edge energy so the box lands on
    structure rather than on a smoothly panning background.

    Ranking by sharp-side energy alone -- the obvious choice -- picks foliage
    every time, which is high-frequency in BOTH frames and so demonstrates
    nothing.

    Scored through summed-area tables. Not faster than the strided python
    loop it replaces (~0.5s vs ~0.4s on a 1176x662 stage) but it evaluates
    every window position rather than one in six, for about the same cost.
    """
    a = np.asarray(blur.convert("L"), dtype=np.float64)
    b = np.asarray(sharp.convert("L"), dtype=np.float64)
    H, W = a.shape
    th, tw = int(H * frac), int(W * frac)

    md = _win_mean(_sat(np.abs(a - b)), th, tw)
    me = _win_mean(_sat(grad_energy(sharp)), th, tw)
    score = md * np.maximum(me, 1e-9) ** 0.30

    # keep the box clear of the two corner labels
    ys, xs = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    score = np.where((ys < H * 0.16) & ((xs < W * 0.32) | (xs > W * 0.68)),
                     score * 0.72, score)

    y, x = np.unravel_index(np.argmax(score), score.shape)
    return ((x + tw / 2) / W, (y + th / 2) / H)


# ----------------------------------------------------------------- motion
def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def seg(t, a, b):
    return 1.0 if b <= a else max(0.0, min(1.0, (t - a) / (b - a)))


def sweep(t, hold=0.18):
    """0 -> 1 -> 0 with a pause at each end, so the eye can compare."""
    if t < hold:
        return 0.0
    if t < 0.5 - hold / 2:
        return ease((t - hold) / (0.5 - 1.5 * hold))
    if t < 0.5 + hold / 2:
        return 1.0
    if t < 1.0 - hold:
        return 1.0 - ease((t - 0.5 - hold / 2) / (0.5 - 1.5 * hold))
    return 0.0


# ----------------------------------------------------------------- chrome
def chrome_overlay(title, subtitle, footer_l, footer_r):
    p = Pen(CW, CH)
    p.rect([0, HEADER_H - 1, CW, HEADER_H + 1], fill=rgba(P["soft"]))
    p.rect([M, 38, M + 7, 112], fill=rgba(P["accent"]))
    p.text((M + 30, 31), title, 72, "bold", rgba(P["head"]))
    w = p.width(title, 72, "bold")
    p.text((M + 30 + w + 30, 62), subtitle, 32, "light", rgba(P["muted"]))
    p.rect([0, CH - 70, CW, CH - 69], fill=rgba(P["soft"]))
    p.text((M, CH - 52), footer_l, 26, "reg", rgba(P["muted"]))
    p.text((CW - M, CH - 52), footer_r, 26, "reg", rgba(P["muted"]), right=True)
    return p.flatten()


def metric(p, x, y, value, unit, label):
    p.text((x, y), value, 72, "mono", rgba(P["head"]))
    w = p.width(value, 72, "mono")
    if unit:
        p.text((x + w + 12, y + 36), unit, 32, "reg", rgba(P["muted"]))
        w += 12 + p.width(unit, 32, "reg")
    p.text((x + 2, y + 78), label.upper(), 24, "reg", rgba(P["muted"]), sp=2.0)
    return x + max(w, p.width(label.upper(), 24, "reg", 2.0))


def pair_overlay(pair, idx, n):
    """Panel borders, captions, measured numbers, tier chip, progress, and
    the part of each leader line that runs outside the stage."""
    p = Pen(CW, CH)
    for y in (PANEL_Y1, PANEL_Y2):
        p.rect([PANEL_X, y, PANEL_X + PANEL_W, y + PANEL_H],
               outline=rgba(P["accent"]), w=3.5)

    for y, lab in ((PANEL_Y1, LEFT_LABEL), (PANEL_Y2, RIGHT_LABEL)):
        p.text((PANEL_X, y - 46), lab + " \u2014 magnified", 32, "bold", rgba(P["ink"]))
        p.text((PANEL_X + PANEL_W, y - 40), "%.2f\u00d7" % pair["zoom"],
               24, "reg", rgba(P["muted"]), right=True)

    # leader elbows: stage edge -> midpoint -> panel edge. Orthogonal routing
    # reads as a pointer; a diagonal dragged across the frame reads as clutter.
    bx, by, bw, bh = pair["box"]
    ey = STAGE_Y + by + bh // 2
    mx = (STAGE_X + STAGE_W + PANEL_X) // 2
    for py in (PANEL_Y1 + PANEL_H // 2, PANEL_Y2 + PANEL_H // 2):
        p.line([(STAGE_X + STAGE_W, ey), (mx, ey)], rgba(P["accent"], 0.55), 3)
        p.line([(mx, ey), (mx, py)], rgba(P["accent"], 0.55), 3)
        p.line([(mx, py), (PANEL_X, py)], rgba(P["accent"], 0.55), 3)

    my = STAGE_Y + STAGE_H + 36
    p.text((M, my - 30), "MEASURED ON THIS PAIR", 24, "reg", rgba(P["muted"]), sp=2.6)
    x = metric(p, M, my, "%.1f" % pair["psnr"], "dB", "PSNR  blur vs sharp")
    x = metric(p, x + 86, my, "%.3f" % pair["ssim"], "", "SSIM")
    metric(p, x + 86, my, "%.0f" % (pair["hf"] * 100), "%", "edge energy lost in crop")

    if pair.get("tier"):
        t = pair["tier"].upper()
        w = p.width(t, 30, "bold", 2.4) + 50
        p.rect([CW - M - w, my - 4, CW - M, my + 56], r=30, fill=rgba(P["accent"]))
        p.text((CW - M - w + 25, my + 10), t, 30, "bold", rgba(P["bg"]), sp=2.4)

    dx = CW - M - 9
    for i in range(n - 1, -1, -1):
        r = 9 if i == idx else 6
        p.ellipse([dx - r, my + 90 - r, dx + r, my + 90 + r],
                  fill=rgba(P["accent"]) if i == idx else rgba(P["soft"]))
        dx -= 36
    lab = "scene %d of %d" % (idx + 1, n)
    p.text((dx + 18, my + 78), lab, 24, "reg", rgba(P["muted"]), right=True, sp=1.6)
    return p.flatten()


def render(bg, pair, s, intro):
    """Compose one finished frame: wipe, then the stage annotations."""
    f = bg.copy()
    x = int(round((1.0 - s) * STAGE_W))
    arr = pair["A"].copy()
    arr[:, x:] = pair["B"][:, x:]
    f.paste(Image.fromarray(arr), (STAGE_X, STAGE_Y))
    return over(f, stage_overlay(pair, s, intro), (STAGE_X, STAGE_Y))


def stage_overlay(pair, s, intro):
    """Per-frame, stage-sized: crop box, leader stub, divider, side labels."""
    p = Pen(STAGE_W, STAGE_H)
    bx, by, bw, bh = pair["box"]
    x = int(round((1.0 - s) * STAGE_W))

    a_box = ease_out(seg(intro, 0.18, 0.55))
    if a_box > 0.01:
        p.rect([bx, by, bx + bw, by + bh], outline=rgba(P["accent"], a_box), w=4)
        p.line([(bx + bw, by + bh // 2), (STAGE_W, by + bh // 2)],
               rgba(P["accent"], 0.35 * a_box), 3)

    if 2 < x < STAGE_W - 2:
        p.line([(x, 0), (x, STAGE_H)], (12, 12, 14, 150), 10)
        p.line([(x, 0), (x, STAGE_H)], (255, 255, 255, 245), 4)
        cy = STAGE_H // 2
        p.ellipse([x - 27, cy - 27, x + 27, cy + 27], fill=(255, 255, 255, 240))
        p.ellipse([x - 27, cy - 27, x + 27, cy + 27], outline=rgba(P["accent"]), w=3)
        for k in (-1, 1):
            p.polygon([(x + k * 16, cy), (x + k * 6, cy - 9), (x + k * 6, cy + 9)],
                      fill=rgba(P["accent"]))

    a_lab = ease_out(seg(intro, 0.0, 0.32))
    for lab, side in ((LEFT_LABEL, "l"), (RIGHT_LABEL, "r")):
        vis = (x / STAGE_W) if side == "l" else (1.0 - x / STAGE_W)
        a = a_lab * (0.32 + 0.68 * max(0.0, min(1.0, vis * 2.2)))
        if a <= 0.01:
            continue
        lw = p.width(lab, 42, "bold") + 48
        lx = 22 if side == "l" else STAGE_W - 22 - lw
        p.rect([lx, 22, lx + lw, 96], r=12, fill=(16, 15, 13, int(215 * a)))
        p.text((lx + 24, 36), lab, 42, "bold", rgba((250, 248, 244), a))
    return p.flatten()


# ------------------------------------------------------------------ pairs
def pair_up(folder, order=None):
    blur, sharp = {}, {}
    for f in sorted(os.listdir(folder)):
        m = re.match(r"(Blur|Sharp)_(.+)\.(jpg|jpeg|png)$", f, re.I)
        if m:
            (blur if m.group(1).lower() == "blur" else sharp)[m.group(2)] = \
                os.path.join(folder, f)
    keys = sorted(set(blur) & set(sharp))
    if order:
        keys = [k for w in [o.strip() for o in order.split(",")] for k in keys if w in k]
    odd = set(blur) ^ set(sharp)
    if odd:
        print("  unmatched, skipped:", ", ".join(sorted(odd)))
    return [(blur[k], sharp[k], k) for k in keys]


def prepare(blur_p, sharp_p, key, tier):
    ob, os_ = load(blur_p), load(sharp_p)
    if ob.size != os_.size:
        os_ = os_.resize(ob.size, Image.LANCZOS)

    a, b = cover(ob, STAGE_W, STAGE_H), cover(os_, STAGE_W, STAGE_H)
    cx, cy = best_crop(a, b)

    bw = int(STAGE_W * CROP_FRAC)
    bh = int(bw * PANEL_H / PANEL_W)
    bx = max(0, min(STAGE_W - bw, int(STAGE_W * cx - bw / 2)))
    by = max(0, min(STAGE_H - bh, int(STAGE_H * cy - bh / 2)))

    # cut the magnified crop from the ORIGINAL file, not the stage copy
    sc = ob.width / STAGE_W
    box = (int(bx * sc), int(by * sc), int((bx + bw) * sc), int((by + bh) * sc))
    cb = ob.crop(box).resize((PANEL_W, PANEL_H), Image.LANCZOS)
    cs = os_.crop(box).resize((PANEL_W, PANEL_H), Image.LANCZOS)

    # thumbnail framing: the crop box widened to 16:9-ish around the subject
    tw_o = min(ob.width, (box[2] - box[0]) * THUMB_ZOOM)
    th_o = tw_o * THUMB_H / THUMB_W
    if th_o > ob.height:
        th_o = ob.height
        tw_o = th_o * THUMB_W / THUMB_H
    tcx, tcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    tx = max(0, min(ob.width - tw_o, tcx - tw_o / 2))
    ty = max(0, min(ob.height - th_o, tcy - th_o / 2))
    tbox = (int(tx), int(ty), int(tx + tw_o), int(ty + th_o))
    tb = ob.crop(tbox).resize((THUMB_W, THUMB_H), Image.LANCZOS)
    ts = os_.crop(tbox).resize((THUMB_W, THUMB_H), Image.LANCZOS)

    na, nb = np.asarray(a), np.asarray(b)
    mad = np.abs(na.astype(np.float64) - nb.astype(np.float64)).mean()
    if mad > 28:
        print("    WARNING: %s / %s differ by mean %.0f/255 -- check they are "
              "the same scene" % (os.path.basename(blur_p), os.path.basename(sharp_p), mad))
    return dict(key=key, tier=tier, A=na, B=nb, box=(bx, by, bw, bh),
                crop_blur=cb, crop_sharp=cs,
                thumb_blur=np.asarray(tb), thumb_sharp=np.asarray(ts),
                zoom=PANEL_W / (box[2] - box[0]),
                psnr=psnr(na, nb), ssim=ssim(na, nb), hf=hf_loss(cb, cs))


def poster_frame(per):
    """Index of the frame whose wipe is nearest half-open. A fixed fraction
    lands on the schedule's end-hold, i.e. on the plain sharp photo."""
    return min(range(per), key=lambda i: abs(sweep(i / per) - 0.5))


def build_thumb(pairs, stem, fps, seconds, crf):
    """The 180x100 publication-list loop: wipe only, no chrome, no type."""
    import imageio.v2 as imageio
    per = max(16, int(round(seconds * fps / len(pairs))))
    w = imageio.get_writer(
        stem + ".mp4", fps=fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", quality=None,
        output_params=["-crf", str(crf), "-preset", "slow",
                       "-movflags", "+faststart", "-an"])
    poster = None
    for pi, p in enumerate(pairs):
        for i in range(per):
            x = int(round((1.0 - sweep(i / per)) * THUMB_W))
            arr = p["thumb_blur"].copy()
            arr[:, x:] = p["thumb_sharp"][:, x:]
            im = Image.fromarray(arr)
            if 1 < x < THUMB_W - 1:
                d = ImageDraw.Draw(im)
                d.line([(x, 0), (x, THUMB_H)], fill=(14, 13, 12), width=3)
                d.line([(x, 0), (x, THUMB_H)], fill=(255, 255, 255), width=1)
            w.append_data(np.asarray(im))
            if pi == 0 and i == poster_frame(per):
                poster = im
        print("  thumb scene %d/%d done" % (pi + 1, len(pairs)))
    w.close()
    poster.save(stem + ".jpg", quality=86, optimize=True, progressive=True)


# ------------------------------------------------------------------- main
def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--order")
    ap.add_argument("--tiers", help="REAL tier per pair, in --order order")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--out", default="iphoneblur")
    ap.add_argument("--title", default="iPhoneBlur")
    ap.add_argument("--subtitle", default="A difficulty-stratified benchmark for "
                                          "consumer-device motion deblurring")
    ap.add_argument("--footer", default="7,400 blur\u2013sharp pairs   \u00b7   "
                                        "iPhone 17 Pro   \u00b7   Easy / Medium / Hard tiers")
    ap.add_argument("--footer-right", default="github.com/C-loud-Nine/iPhoneBlur")
    ap.add_argument("--crf", type=int, default=19,
                    help="18 = near-lossless and larger, 28 = small and soft")
    ap.add_argument("--no-abstract", action="store_true")
    ap.add_argument("--no-thumb", action="store_true")
    ap.add_argument("--thumb-out", default="thumb_iphoneblur")
    ap.add_argument("--thumb-seconds", type=float, default=6.0)
    a = ap.parse_args()
    P = DARK if a.dark else LIGHT

    pairs_p = pair_up(a.folder, a.order)
    if not pairs_p:
        sys.exit("no Blur_/Sharp_ pairs found in " + a.folder)
    tiers = [t.strip() for t in a.tiers.split(",")] if a.tiers else []

    print("%d pair(s):" % len(pairs_p))
    pairs = []
    for i, (bp, sp, k) in enumerate(pairs_p):
        p = prepare(bp, sp, k, tiers[i] if i < len(tiers) else None)
        pairs.append(p)
        print("    %-22s PSNR %5.1f dB   SSIM %.3f   edge loss %2.0f%%   crop %.2fx"
              % (k, p["psnr"], p["ssim"], p["hf"] * 100, p["zoom"]))

    base = Image.new("RGB", (CW, CH), P["bg"])
    ch_ov = chrome_overlay(a.title, a.subtitle, a.footer, a.footer_right)
    chrome = over(base, ch_ov)

    bgs = []
    for i, p in enumerate(pairs):
        c = base.copy()
        for box in ([STAGE_X, STAGE_Y, STAGE_X + STAGE_W, STAGE_Y + STAGE_H],
                    [PANEL_X, PANEL_Y1, PANEL_X + PANEL_W, PANEL_Y1 + PANEL_H],
                    [PANEL_X, PANEL_Y2, PANEL_X + PANEL_W, PANEL_Y2 + PANEL_H]):
            soft_shadow(c, box)
        c.paste(p["crop_blur"], (PANEL_X, PANEL_Y1))
        c.paste(p["crop_sharp"], (PANEL_X, PANEL_Y2))
        bgs.append(over(over(c, ch_ov), pair_overlay(p, i, len(pairs))))

    per = max(24, int(round(a.seconds * a.fps / len(pairs))))
    fade = max(4, int(a.fps * 0.30))
    print("%d frames, %dx%d, %.1fs @ %d fps"
          % (per * len(pairs), CW, CH, per * len(pairs) / a.fps, a.fps))

    try:
        import imageio.v2 as imageio
    except ImportError:
        sys.exit("pip install imageio imageio-ffmpeg")

    stem = os.path.join(a.folder, a.out)
    # H.264 only: it is the one codec every browser and every reviewer's
    # machine plays, and +faststart lets it start before it has downloaded.
    w = imageio.get_writer(
        stem + ".mp4", fps=a.fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", quality=None,
        output_params=["-crf", str(a.crf), "-preset", "slow", "-profile:v", "high",
                       "-movflags", "+faststart", "-an"])

    poster = None
    for pi, p in enumerate(pairs):
        for i in range(per):
            t = i / per
            # intro is progress through THIS scene, not the whole clip, so
            # every scene animates in instead of snapping at the first cut
            f = render(bgs[pi], p, sweep(t), t)
            if i < fade:
                f = Image.blend(chrome, f, i / fade)
            elif i >= per - fade:
                f = Image.blend(chrome, f, (per - i) / fade)
            w.append_data(np.asarray(f))       # streamed, never accumulated
            if pi == 0 and i == poster_frame(per):
                poster = f
        print("  scene %d/%d done" % (pi + 1, len(pairs)))
    w.close()

    if poster:
        poster.save(stem + "_poster.jpg", quality=88, optimize=True, progressive=True)
    if not a.no_abstract:
        render(bgs[0], pairs[0], 0.5, 1.0).save(stem + "_abstract.png", optimize=True)

    if not a.no_thumb:
        build_thumb(pairs, os.path.join(a.folder, a.thumb_out), a.fps,
                    a.thumb_seconds, a.crf + 4)

    print()
    tstem = os.path.join(a.folder, a.thumb_out)
    for f in (stem + ".mp4", stem + "_poster.jpg", stem + "_abstract.png",
              tstem + ".mp4", tstem + ".jpg"):
        if os.path.exists(f):
            print("  %-30s %8.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))


if __name__ == "__main__":
    main()
