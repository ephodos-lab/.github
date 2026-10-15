"""Generate the Ephodos Lab brand assets as SVG.

The mark is one formula. Take a lattice that is uniform in log-polar coordinates,

    r = phi^(-j/2),   theta = 2 pi k / N + j d / phi^2     (arm k = 0..N-1, ring j = 0..13),

where phi is the golden ratio and d = 2 pi / N the gap between arms: two rings make one
phi, and each ring is turned by 1/phi^2 of the gap, so every arm is a logarithmic spiral.
Lay it on the funnel z = ln r and look at it obliquely, so depth slides each ring along
the major axis:

    major = r cos theta + a ln(1/r),   minor = eps r sin theta,   a just under r_min.

The throat then sits off centre and its far wall is seen almost edge-on, a dense crescent.

The dots fade along a band centred on another logarithmic spiral, theta = theta0 - pi u with
u = ln(r / r_min) / ln(1 / r_min): it leaves the throat at theta0 and turns half a turn, the
same way as the arms, on its way out to the rim. With psi a point's angle from the band,

    psi = theta - theta0 + pi u,   b = ((1 + cos psi) / 2)^(sharp r^(1/3)),   m = 1 - (1 - c u) b,

so the band is wide and deep at the throat and narrows, but stays deep, out to the rim. The
dot radius is rho r^gamma m (the outermost dot of each arm phi times larger), and each arm is
a stroke whose half-width is a tenth of rho r^gamma times m^4, so the band shows as a white
spiral that runs from the throat through the outermost ring.

The wordmark sets EPHODOS LAB in Cormorant Garamond with the first O drawn as a crescent;
the tagline is Jost.

    python build.py --fonts DIR --out DIR

DIR must contain CormorantGaramond[wght].ttf and Jost[wght].ttf, both under the SIL Open
Font License, from https://github.com/google/fonts (ofl/cormorantgaramond and ofl/jost).
Requires numpy and fontTools.
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

PHI = (1 + 5 ** 0.5) / 2

VORTEX = dict(
    N=32,                       # arms
    J=13,                       # rings inside the rim; ring j has radius phi^(-j/2)
    turn=1 / PHI ** 2,          # each ring is turned by this fraction of the gap between arms
    lean=0.98,                  # depth slides ring r by lean r_min ln(1/r) along the major axis (< 1: no fold)
    eps=2 / 3,                  # foreshortening of the minor axis
    beta=math.radians(143),     # direction of the major axis on screen (y down), toward the throat
    rho=0.02, gamma=0.5,        # dot radius rho r^gamma before the fade
    tip=PHI,                    # the outermost dot of each arm is phi times larger
    stroke=0.1,                 # half-width of an arm's stroke, as a fraction of its dot radius
    theta0=math.radians(125),   # where the fading band leaves the throat
    sweep=math.pi,              # how far the band turns from the throat to the rim
    sharp=13.0, sharp_pow=1 / 3,  # band = ((1 + cos psi) / 2)^(sharp r^sharp_pow): wide at the throat, narrow at the rim
    c=0.15,                     # the band's depth falls from 1 at the throat to 1 - c at the rim
    min_fade=0.04,              # dots faded below this are left out
)


def vortex(v=VORTEX, sub=6):
    """The arms of the mark: dot centres and radii, and each arm's centre line with its stroke half-width."""
    q, J, N = PHI ** 0.5, v["J"], v["N"]
    r_min = q ** -J
    a, L = v["lean"] * r_min, math.log(1 / r_min)
    d = 2 * math.pi / N * v["turn"]
    cb, sb = math.cos(v["beta"]), math.sin(v["beta"])
    arms = []
    for k in range(N):
        def at(j):
            r = q ** -np.asarray(j, float)
            t = 2 * math.pi * k / N + np.asarray(j, float) * d
            major, minor = r * np.cos(t) + a * np.log(1 / r), v["eps"] * r * np.sin(t)
            u = np.log(r / r_min) / L
            b = ((1 + np.cos(t - v["theta0"] + v["sweep"] * u)) / 2) ** (v["sharp"] * r ** v["sharp_pow"])
            m = 1 - (1 - v["c"] * u) * b
            return major * cb - minor * sb, major * sb + minor * cb, v["rho"] * r ** v["gamma"], m
        x, y, rad, m = at(np.arange(J + 1))
        rad = rad * m
        rad[0] *= v["tip"]
        vis = m >= v["min_fade"]
        lx, ly, lr, lm = at(np.linspace(0, J, J * sub + 1))
        arms.append({"x": x[vis], "y": y[vis], "r": rad[vis], "line": (lx, ly, v["stroke"] * lr * lm ** 4)})
    return arms


def mark_bounds(arms):
    xs = np.concatenate([np.r_[a["x"] - a["r"], a["x"] + a["r"]] for a in arms])
    ys = np.concatenate([np.r_[a["y"] - a["r"], a["y"] + a["r"]] for a in arms])
    return xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()


def _ribbon(x, y, hw):
    """Outline of a stroke along (x, y) with half-width hw, as one closed polygon."""
    dx, dy = np.gradient(x), np.gradient(y)
    n = np.hypot(dx, dy)
    nx, ny = -dy / n * hw, dx / n * hw
    return np.r_[np.c_[x + nx, y + ny], np.c_[x - nx, y - ny][::-1]]


def mark_svg(arms, box, color, dot_scale=1.0, stroke_scale=1.0, min_px=0.4, min_stroke_px=0.06):
    """SVG group drawing the mark scaled into box = (x, y, width, height), centred.

    dot_scale and stroke_scale thicken the dots and strokes (a heavier cut for small sizes);
    dots smaller than min_px and strokes thinner than min_stroke_px after scaling are left out."""
    x0, y0, w, h = mark_bounds(arms)
    s = min(box[2] / w, box[3] / h)
    ox, oy = box[0] + (box[2] - w * s) / 2 - x0 * s, box[1] + (box[3] - h * s) / 2 - y0 * s
    f = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
    strokes = []
    for a in arms:
        lx, ly, hw = a["line"]
        hw = hw * s * stroke_scale
        idx = np.flatnonzero(2 * hw >= min_stroke_px)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) > 1:
                poly = _ribbon(ox + lx[run] * s, oy + ly[run] * s, hw[run])
                strokes.append("M" + "L".join(f"{f(px)} {f(py)}" for px, py in poly) + "Z")
    dots = "".join(f'<circle cx="{f(ox + px * s)}" cy="{f(oy + py * s)}" r="{f(pr * s * dot_scale)}"/>'
                   for a in arms for px, py, pr in zip(a["x"], a["y"], a["r"]) if pr * s * dot_scale >= min_px)
    return f'<g fill="{color}"><path d="{"".join(strokes)}"/>{dots}</g>'


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
        files[name] = svg_doc(1024, 1024, mark_svg(arms, (92, 92, 840, 840), ink, dot_scale=1.4, stroke_scale=1.6,
                                                    min_px=1.2), "Ephodos Lab", bg=bg)
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
