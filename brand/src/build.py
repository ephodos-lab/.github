"""Generate the Ephodos Lab brand assets as SVG.

The mark is a dotted vortex: 32 arms of dots spiral into an off-centre throat,
seen at an angle. Its geometry is fully parametric; the parameters below were
measured from the original concept artwork. The wordmark sets EPHODOS LAB in
Cormorant Garamond with the first O drawn as a crescent; the tagline is Jost.

    python build.py --fonts DIR --out DIR

DIR must contain CormorantGaramond[wght].ttf and Jost[wght].ttf, both under the
SIL Open Font License, from https://github.com/google/fonts (ofl/cormorantgaramond
and ofl/jost). Requires numpy and fontTools.
"""
import argparse
import functools
import math
import os

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

INK, PAPER, IVORY = "#0B1220", "#FFFFFF", "#EAE3D2"

# ---------------------------------------------------------------- the mark

# Arm curve: a ring parameter t runs from the throat (0) to the rim (1). Each ring is an
# ellipse whose centre slides from the throat centre H to the rim centre C as t grows.
CURVE = dict(k=0.71, m=1.02, k_throat=1.6, p_throat=6.0,          # twist: gentle overall, strong at the throat
             mu=3.02, nu=1.20, rt=0.068,                           # centre slide, radius growth, throat radius
             C=(660.0, 475.0), H=(613.0, 496.0),                   # rim and throat centres
             A=314.0, B=238.0, beta=142.5, lam=0.6)                # rim semi-axes, major-axis angle, roundness easing
N_ARMS = 32
ARM_ANGLES = (4.2771, 7.6065, 180.2156, 5.2225, 6.9226)          # a0, c1, p1, c2, p2: alpha_i = a0 + phi + c1 sin(phi-p1) + c2 sin(2phi-p2)
ARM_LENGTH = (0.9638, 0.009, 0.0078, 0.0405, -0.0005, 0.0072, 0.0085, 0.1131, 291.573, 19.5836)
TIP_RADIUS = (7.9233, -1.315, -0.3659, 1.8679, -1.1191, 0.3946, 0.0608, 0.7581, 307.2723, 11.5716)
DOT_RADIUS = (5.4219, -1.0844, 0.5499, 0.1165, -0.6628, 0.8446, -0.0307, 0.3309, 0.0184, -0.0322, 0.4364, 314.7678, 26.6343)
SPACING = (1.1219, 0.0287)                                        # s_j / s_(j+1) = a + b s_j
TIP_GAP = 1.20                                                   # the bead at the end sits a little further out
THROAT = dict(floor=1.3, ring=(25.0, 195.0), void=(195.0, 245.0), void_t=0.34)


def _harmonics(c, deg, tail=None):
    a = math.radians(deg)
    v = c[0] + sum(c[2 * k - 1] * math.cos(k * a) + c[2 * k] * math.sin(k * a) for k in range(1, (len(c) - 1) // 2 + 1))
    if tail is not None:
        depth, centre, width = tail
        d = (deg - centre + 180) % 360 - 180
        v *= 1 - depth * math.exp(-(d / width) ** 2)
    return v


def _arm_xy(alpha, t):
    c = CURVE
    t = np.asarray(t, float)
    out = np.clip(1 - t, 0, None)
    th = math.radians(alpha) + c["k"] * out ** c["m"] + c["k_throat"] * out ** c["p_throat"]
    rho = c["rt"] + (1 - c["rt"]) * t ** c["nu"]
    roundness = 1.0 + (c["B"] / c["A"] - 1.0) * t ** c["lam"]
    cx = c["H"][0] + (c["C"][0] - c["H"][0]) * t ** c["mu"]
    cy = c["H"][1] + (c["C"][1] - c["H"][1]) * t ** c["mu"]
    u, v = c["A"] * rho * np.cos(th), c["A"] * roundness * rho * np.sin(th)
    b = math.radians(c["beta"])
    return cx + u * math.cos(b) - v * math.sin(b), cy + u * math.sin(b) + v * math.cos(b)


def _dot_radius(alpha, s):
    p = DOT_RADIUS
    a = math.radians(alpha)
    four = lambda c: c[0] + c[1] * math.cos(a) + c[2] * math.sin(a) + c[3] * math.cos(2 * a) + c[4] * math.sin(2 * a)
    d = (alpha - p[11] + 180) % 360 - 180
    return four(p[:5]) * (1 - p[10] * math.exp(-(d / p[12]) ** 2)) * np.asarray(s) ** four(p[5:10])


def vortex(t_min=0.004, omit_r=0.8):
    """Return the arms of the mark as dicts of dot centres, radii and a hairline, in source pixel units."""
    a0, c1, p1, c2, p2 = ARM_ANGLES
    arms = []
    for i in range(N_ARMS):
        phi = 360.0 * i / N_ARMS
        alpha = (a0 + phi + c1 * math.sin(math.radians(phi - p1)) + c2 * math.sin(math.radians(2 * phi - p2))) % 360
        t_end = _harmonics(ARM_LENGTH[:7], alpha, ARM_LENGTH[7:])
        ts, t = [t_end], t_end / TIP_GAP
        while t >= t_min:
            ts.append(t)
            t /= SPACING[0] + SPACING[1] * t
        ts = np.array(ts)
        x, y = _arm_xy(alpha, ts)
        r = _dot_radius(alpha, ts)
        r[0] = _harmonics(TIP_RADIUS[:7], alpha, TIP_RADIUS[7:])
        # the throat: a dense ring on the near wall, a clear opening on the far side
        ang = np.degrees(np.arctan2(y - CURVE["H"][1], x - CURVE["H"][0])) % 360
        wall = (ang >= THROAT["ring"][0]) & (ang <= THROAT["ring"][1])
        r = np.where((ts < 0.3) & wall, np.maximum(r, THROAT["floor"] * (0.55 + 0.45 * ts / 0.3)), r)
        void = (ang > THROAT["void"][0]) & (ang < THROAT["void"][1])
        r = np.where(void, r * np.clip(ts / THROAT["void_t"], 0, 1) ** 2, r)
        keep = r >= omit_r
        ts, x, y, r = ts[keep], x[keep], y[keep], r[keep]
        if not len(ts):
            continue
        lt = np.linspace(ts.min(), ts.max(), 160)
        arms.append({"x": x, "y": y, "r": r, "line": _arm_xy(alpha, lt)})
    return arms


def mark_bounds(arms):
    xs = np.concatenate([np.r_[a["x"] - a["r"], a["x"] + a["r"]] for a in arms])
    ys = np.concatenate([np.r_[a["y"] - a["r"], a["y"] + a["r"]] for a in arms])
    return xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()


def mark_svg(arms, box, color, hairline=1.05, hairline_opacity=0.75, dot_scale=1.0, min_px=0.0):
    """SVG group drawing the mark scaled into box = (x, y, width, height), centred.

    dot_scale enlarges every dot (a heavier cut for small sizes); dots smaller than min_px
    after scaling are left out."""
    x0, y0, w, h = mark_bounds(arms)
    s = min(box[2] / w, box[3] / h)
    ox, oy = box[0] + (box[2] - w * s) / 2 - x0 * s, box[1] + (box[3] - h * s) / 2 - y0 * s
    f = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
    lines = "".join('<polyline points="%s"/>' % " ".join(f"{f(ox + px * s)},{f(oy + py * s)}" for px, py in zip(*a["line"])) for a in arms)
    dots = "".join(f'<circle cx="{f(ox + px * s)}" cy="{f(oy + py * s)}" r="{f(pr * s * dot_scale)}"/>'
                   for a in arms for px, py, pr in zip(a["x"], a["y"], a["r"]) if pr * s * dot_scale >= min_px)
    return (f'<g fill="none" stroke="{color}" stroke-width="{f(hairline * s)}" stroke-opacity="{hairline_opacity}" '
            f'stroke-linecap="round" stroke-linejoin="round">{lines}</g><g fill="{color}">{dots}</g>')


# ------------------------------------------------------------- the wordmark

@functools.lru_cache(None)
def _font(path, wght):
    f = TTFont(path)
    return instancer.instantiateVariableFont(f, {"wght": wght}) if "fvar" in f else f


def _cap_scale(path, wght, cap):
    f = _font(path, wght)
    gs, cm = f.getGlyphSet(), f.getBestCmap()
    bp = BoundsPen(gs)
    gs[cm[ord("H")]].draw(bp)
    return cap / bp.bounds[3]


def _glyph(path, wght, ch, cap, x, baseline, align="ink"):
    """Path of one glyph; x is its ink left edge (align="ink") or its origin (align="advance")."""
    f = _font(path, wght)
    gs, cm = f.getGlyphSet(), f.getBestCmap()
    s = _cap_scale(path, wght, cap)
    name = cm[ord(ch)]
    bp = BoundsPen(gs)
    gs[name].draw(bp)
    adv = f["hmtx"][name][0] * s
    if bp.bounds is None:
        return "", 0.0, adv
    x0, _, x1, _ = bp.bounds
    tx = x - x0 * s if align == "ink" else x
    pen = SVGPathPen(gs, ntos=lambda v: ("%.2f" % v).rstrip("0").rstrip("."))
    gs[name].draw(TransformPen(pen, (s, 0, 0, -s, tx, baseline)))
    return pen.getCommands(), (x1 - x0) * s, adv


# Crescent O, measured from the concept artwork (units of cap height): outer radius R, inner radius r,
# inner circle shifted by d toward GAP degrees (screen coordinates, y down).
CRESCENT = dict(R=0.5407, r=0.4667, d=0.0839, gap=172.0)


def _crescent(cap, x, baseline):
    R, r, d = (CRESCENT[k] * cap for k in ("R", "r", "d"))
    g = math.radians(CRESCENT["gap"])
    ix, iy = d * math.cos(g), d * math.sin(g)
    a = (R * R - r * r + d * d) / (2 * d)
    h = math.sqrt(R * R - a * a)
    mx, my = a * ix / d, a * iy / d
    tips = [(mx + h * iy / d, my - h * ix / d), (mx - h * iy / d, my + h * ix / d)]
    ang = lambda p: math.atan2(p[1], p[0])
    start, end = sorted(tips, key=lambda p: (ang(p) - g) % (2 * math.pi))
    span = (ang(end) - ang(start)) % (2 * math.pi)
    xs = [R * math.cos(ang(start) + span * k / 720) for k in range(721)]
    x_min = min(xs + [start[0], end[0]])
    cx, cy = x - x_min, baseline - cap / 2
    P = lambda p: "%.2f %.2f" % (cx + p[0], cy + p[1])
    return (f"M{P(start)} A{R:.2f} {R:.2f} 0 1 1 {P(end)} A{r:.2f} {r:.2f} 0 1 0 {P(start)} Z", max(xs) - x_min)


def wordmark(serif, cap, x, baseline, heavy=530, light=300, gap=0.185, word_gap=0.73):
    """EPHODOS LAB as path data; returns (paths, width). Letters are spaced by equal gaps between their ink."""
    paths, cur = [], x
    for i, ch in enumerate("EPHODOS"):
        d, w = _crescent(cap, cur, baseline) if i == 3 else _glyph(serif, heavy, ch, cap, cur, baseline)[:2]
        paths.append(d)
        cur += w + gap * cap
    cur += (word_gap - gap) * cap
    for ch in "LAB":
        d, w, _ = _glyph(serif, light, ch, cap, cur, baseline)
        paths.append(d)
        cur += w + gap * cap
    return paths, cur - gap * cap - x


def spaced_text(font, wght, text, cap, x, baseline, tracking):
    """Uppercase text set by advance widths plus tracking (in units of cap height); returns (paths, width)."""
    paths, cur = [], x
    for ch in text:
        d, _, adv = _glyph(font, wght, ch, cap, cur, baseline, align="advance")
        if d:
            paths.append(d)
        cur += adv + tracking * cap
    return paths, cur - tracking * cap - x


def text_width(font, wght, text, cap, tracking):
    return spaced_text(font, wght, text, cap, 0, 0, tracking)[1]


# ------------------------------------------------------------------ assets

def svg_doc(w, h, body, title, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{title}</title>{rect}{body}</svg>\n'


def build(fonts, out):
    serif = os.path.join(fonts, "CormorantGaramond[wght].ttf")
    sans = os.path.join(fonts, "Jost[wght].ttf")
    arms = vortex()
    tag1, tag2, tag_one = "OPEN TOOLS FOR AUTOMATED", "MATHEMATICAL RESEARCH", "OPEN TOOLS FOR AUTOMATED MATHEMATICAL RESEARCH"
    files = {}

    def paths(ps, color):
        return f'<g fill="{color}">' + "".join(f'<path d="{d}"/>' for d in ps) + "</g>"

    for suffix, ink in (("", INK), ("-on-dark", IVORY)):
        # mark alone, transparent ground
        files[f"mark{suffix}.svg"] = svg_doc(1024, 1024, mark_svg(arms, (32, 32, 960, 960), ink), "Ephodos Lab")
        # wordmark alone
        cap = 100.0
        ps, w = wordmark(serif, cap, 8, 8 + cap * (0.5 + CRESCENT["R"]) + cap / 2)
        H = cap * 2 * CRESCENT["R"] + 16
        files[f"wordmark{suffix}.svg"] = svg_doc(round(w + 16), round(H), paths(ps, ink), "Ephodos Lab")
        # square lockup: mark, wordmark, two-line tagline (proportions of the concept artwork)
        S = 1254
        body = mark_svg(arms, (338, 209, 578, 532), ink)
        cap = 84.0
        wm_w = wordmark(serif, cap, 0, 0)[1]
        ps, _ = wordmark(serif, cap, (S - wm_w) / 2, 890)
        body += paths(ps, ink)
        tcap, track = 22.0, 0.42
        for line, base in ((tag1, 972), (tag2, 1023)):
            lw = text_width(sans, 300, line, tcap, track)
            body += paths(spaced_text(sans, 300, line, tcap, (S - lw) / 2, base, track)[0], ink)
        files[f"lockup{suffix}.svg"] = svg_doc(S, S, body, "Ephodos Lab", bg=None)

    # avatar: the mark on white
    for name, bg, ink in (("avatar.svg", PAPER, INK), ("avatar-on-dark.svg", INK, IVORY)):
        files[name] = svg_doc(1024, 1024, mark_svg(arms, (92, 92, 840, 840), ink, hairline=2.2, hairline_opacity=0.9,
                                                    dot_scale=1.75, min_px=2.2), "Ephodos Lab", bg=bg)
    # banner for the organisation profile
    W, Hh = 1600, 480
    _, _, mw, mh = mark_bounds(arms)
    mark_h = 340.0
    mark_w = mark_h * mw / mh
    cap, gap_x = 74.0, 96.0
    wm_w = wordmark(serif, cap, 0, 0)[1]
    x0 = (W - (mark_w + gap_x + wm_w)) / 2
    body = mark_svg(arms, (x0, (Hh - mark_h) / 2, mark_w, mark_h), INK)
    tx = x0 + mark_w + gap_x
    body += paths(wordmark(serif, cap, tx, 236)[0], INK)
    tcap, track = 17.0, 0.40
    tw = text_width(sans, 300, tag_one, tcap, track)
    if tw > wm_w:
        tcap *= wm_w / tw
    body += paths(spaced_text(sans, 300, tag_one, tcap, tx + 2, 306, track)[0], INK)
    files["banner.svg"] = svg_doc(W, Hh, body, "Ephodos Lab: open tools for automated mathematical research", bg=PAPER)

    os.makedirs(out, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return sorted(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fonts", required=True, help="directory with CormorantGaramond[wght].ttf and Jost[wght].ttf")
    ap.add_argument("--out", required=True, help="output directory for the SVG files")
    a = ap.parse_args()
    for n in build(a.fonts, a.out):
        print(n)
