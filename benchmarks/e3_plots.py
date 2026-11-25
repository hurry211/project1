"""Dependency-free publication SVG figure generator for E3 benchmarks."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable


_PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")


def _ticks(lo: float, hi: float, log_scale: bool, count: int = 5) -> list[float]:
    if log_scale:
        p0 = math.floor(math.log10(max(lo, 1e-12)))
        p1 = math.ceil(math.log10(max(hi, 1e-12)))
        return [10.0 ** p for p in range(p0, p1 + 1)]
    span = max(hi - lo, 1e-12)
    step = 10.0 ** math.floor(math.log10(span / max(count - 1, 1)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if span / (step * mult) <= count:
            step *= mult
            break
    start = math.floor(lo / step) * step
    vals = []
    cur = start
    while cur <= hi + 0.01 * step:
        vals.append(cur)
        cur += step
    return vals


def _fmt_num(v: float, log_scale: bool) -> str:
    if log_scale and v > 0:
        exp = math.log10(v)
        if abs(exp - round(exp)) < 1e-6:
            r = int(round(exp))
            if -2 <= r <= 3:
                return f"{v:g}"
            return f"1e{r}"
    if abs(v) >= 1000 or (0 < abs(v) < 1e-2):
        return f"{v:.2g}"
    return f"{v:.3g}"


def render_line_chart(
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict],
    x_log: bool = True,
    y_log: bool = True,
    width: int = 760,
    height: int = 440,
) -> Path:
    """Render a multi-series line/scatter plot to standalone SVG."""
    ml, mr, mt, mb = 84, 32, 48, 66
    pw, ph = width - ml - mr, height - mt - mb

    xs = [float(x) for s in series for x in s["x"] if x is not None and math.isfinite(x)]
    ys = [float(y) for s in series for y in s["y"] if y is not None and math.isfinite(y)]
    if x_log:
        xs = [x for x in xs if x > 0]
    if y_log:
        ys = [y for y in ys if y > 0]
    if not xs or not ys:
        raise ValueError("cannot plot empty series")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if x_log:
        lx0, lx1 = math.log10(xmin) - 0.12, math.log10(xmax) + 0.12
    else:
        pad = 0.08 * max(xmax - xmin, 1.0)
        lx0, lx1 = xmin - pad, xmax + pad
    if y_log:
        ly0, ly1 = math.log10(ymin) - 0.15, math.log10(ymax) + 0.18
    else:
        pad = 0.12 * max(ymax - ymin, 1.0)
        ly0, ly1 = max(0.0, ymin - 0.05 * ymax) if ymin >= 0 else ymin - pad, ymax + pad

    def map_x(x: float) -> float:
        tx = math.log10(x) if x_log else x
        return ml + (tx - lx0) / max(lx1 - lx0, 1e-12) * pw

    def map_y(y: float) -> float:
        ty = math.log10(y) if y_log else y
        return mt + ph - (ty - ly0) / max(ly1 - ly0, 1e-12) * ph

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="16" font-weight="600" fill="#111">{html.escape(title)}</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    for xv in _ticks(10**lx0 if x_log else lx0, 10**lx1 if x_log else lx1, x_log):
        if (x_log and (10**lx0 <= xv <= 10**lx1)) or (not x_log and (lx0 <= xv <= lx1)):
            px = map_x(xv)
            elems.append(f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
            elems.append(f'<text x="{px:.1f}" y="{mt+ph+20}" text-anchor="middle" font-family="monospace" font-size="11" fill="#475569">{html.escape(_fmt_num(xv, x_log))}</text>')
    for yv in _ticks(10**ly0 if y_log else ly0, 10**ly1 if y_log else ly1, y_log):
        if (y_log and (10**ly0 <= yv <= 10**ly1)) or (not y_log and (ly0 <= yv <= ly1)):
            py = map_y(yv)
            elems.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml+pw}" y2="{py:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
            elems.append(f'<text x="{ml-8}" y="{py+4:.1f}" text-anchor="end" font-family="monospace" font-size="11" fill="#475569">{html.escape(_fmt_num(yv, y_log))}</text>')

    elems.append(f'<text x="{ml+pw/2:.1f}" y="{height-16}" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="13" fill="#1e293b">{html.escape(xlabel)}</text>')
    elems.append(f'<text x="20" y="{mt+ph/2:.1f}" transform="rotate(-90 20 {mt+ph/2:.1f})" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="13" fill="#1e293b">{html.escape(ylabel)}</text>')

    for idx, s in enumerate(series):
        color = s.get("color", _PALETTE[idx % len(_PALETTE)])
        dash = ' stroke-dasharray="6,4"' if s.get("dashed") else ""
        pts = [(x, y) for x, y in zip(s["x"], s["y"]) if x is not None and y is not None and (not x_log or x > 0) and (not y_log or y > 0)]
        if not pts:
            continue
        poly = " ".join(f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in pts)
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3"{dash} points="{poly}"/>')
        for x, y in pts:
            elems.append(f'<circle cx="{map_x(x):.1f}" cy="{map_y(y):.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1"/>')

    # Legend box top-left or bottom-right
    lx_leg, ly_leg = ml + 14, mt + 12
    elems.append(f'<rect x="{lx_leg}" y="{ly_leg}" width="250" height="{14 + 20 * len(series)}" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0"/>')
    for idx, s in enumerate(series):
        color = s.get("color", _PALETTE[idx % len(_PALETTE)])
        cy = ly_leg + 16 + 20 * idx
        dash = ' stroke-dasharray="5,3"' if s.get("dashed") else ""
        elems.append(f'<line x1="{lx_leg+10}" y1="{cy}" x2="{lx_leg+34}" y2="{cy}" stroke="{color}" stroke-width="2.5"{dash}/>')
        elems.append(f'<text x="{lx_leg+42}" y="{cy+4}" font-family="system-ui,Segoe UI,Arial" font-size="11.5" fill="#0f172a">{html.escape(s["label"])}</text>')

    elems.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elems) + "\n", encoding="utf-8")
    return output_path


def render_bar_comparison(
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    bars: list[dict],
    width: int = 760,
    height: int = 420,
) -> Path:
    """Render a strategy comparison bar chart with optional 95% CI whiskers."""
    ml, mr, mt, mb = 78, 30, 48, 86
    pw, ph = width - ml - mr, height - mt - mb
    vals = [float(b.get("hi", b["value"])) for b in bars if b.get("value") is not None]
    ymax = max(vals) * 1.18 if vals else 1.0

    def map_y(y: float) -> float:
        return mt + ph - (y / max(ymax, 1e-12)) * ph

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="16" font-weight="600" fill="#111">{html.escape(title)}</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e1"/>',
    ]
    for yv in _ticks(0.0, ymax, False):
        if 0 <= yv <= ymax:
            py = map_y(yv)
            elems.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml+pw}" y2="{py:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
            elems.append(f'<text x="{ml-8}" y="{py+4:.1f}" text-anchor="end" font-family="monospace" font-size="11" fill="#475569">{yv:.3g}</text>')

    elems.append(f'<text x="20" y="{mt+ph/2:.1f}" transform="rotate(-90 20 {mt+ph/2:.1f})" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="13" fill="#1e293b">{html.escape(ylabel)}</text>')

    slot = pw / max(len(bars), 1)
    bw = min(64.0, slot * 0.56)
    for i, b in enumerate(bars):
        cx = ml + slot * (i + 0.5)
        color = b.get("color", _PALETTE[i % len(_PALETTE)])
        val = float(b["value"])
        top = map_y(val)
        bot = map_y(0.0)
        elems.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(bot-top, 1):.1f}" fill="{color}" fill-opacity="0.88" rx="3"/>')
        if b.get("lo") is not None and b.get("hi") is not None:
            ylo, yhi = map_y(float(b["lo"])), map_y(float(b["hi"]))
            elems.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="#0f172a" stroke-width="1.6"/>')
            elems.append(f'<line x1="{cx-6:.1f}" y1="{ylo:.1f}" x2="{cx+6:.1f}" y2="{ylo:.1f}" stroke="#0f172a" stroke-width="1.6"/>')
            elems.append(f'<line x1="{cx-6:.1f}" y1="{yhi:.1f}" x2="{cx+6:.1f}" y2="{yhi:.1f}" stroke="#0f172a" stroke-width="1.6"/>')
        elems.append(f'<text x="{cx:.1f}" y="{min(top-6, map_y(float(b.get("hi", val)))-6):.1f}" text-anchor="middle" font-family="monospace" font-size="11" fill="#0f172a">{val:.3g}</text>')
        elems.append(f'<text x="{cx:.1f}" y="{mt+ph+20}" text-anchor="middle" font-family="system-ui,Segoe UI,Arial" font-size="11.5" fill="#1e293b">{html.escape(b["label"])}</text>')
        if b.get("sublabel"):
            elems.append(f'<text x="{cx:.1f}" y="{mt+ph+36}" text-anchor="middle" font-family="monospace" font-size="10" fill="#64748b">{html.escape(b["sublabel"])}</text>')

    elems.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elems) + "\n", encoding="utf-8")
    return output_path
