"""Appendix D's figures, written as SVG from the stack that is already pinned.

Appendix D names four `.png` files. The pinned stack (spec §2.3) has no
plotting library, and adding matplotlib for four charts would be a substitution
requiring an ADR -- so these are SVG, written from the numbers. See ADR-0024
for the reasoning and for what is given up.

Purity is the reason this is testable at all: every function takes measured
values and returns a string. There is no canvas, no global figure state and no
file handle, so a test asserts on the document rather than on whether a file
appeared.

Nothing here computes a statistic. If a figure needs a number, the number is
passed in, already measured -- a chart that quietly recomputes its own data is
a second implementation of the metric, and the two will eventually disagree in
the direction nobody checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

from cascade.eval.schema import CalibrationReport, Comparison

__all__ = [
    "Box",
    "convergence_svg",
    "forest_svg",
    "reliability_svg",
    "scatter_svg",
]

# A palette that survives greyscale printing and the two most common forms of
# colour blindness. A reliability diagram whose ideal line and measured curve
# are distinguishable only by hue is a diagram half the readers cannot read.
_INK = "#1a1a1a"
_GRID = "#d5d5d5"
_ACCENT = "#0b6fa4"
_WARN = "#b3541e"
_MUTED = "#8a8a8a"

_FONT = "font-family='ui-sans-serif,system-ui,-apple-system,Helvetica,Arial,sans-serif'"


@dataclass(frozen=True, slots=True)
class Box:
    """A plotting rectangle in SVG user units, and the data range it shows."""

    x: float
    y: float
    width: float
    height: float
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float

    def sx(self, value: float) -> float:
        """Map a data x onto the canvas, clamped to the box."""
        if self.x_hi == self.x_lo:
            return self.x + self.width / 2.0
        t = (value - self.x_lo) / (self.x_hi - self.x_lo)
        return self.x + min(max(t, 0.0), 1.0) * self.width

    def sy(self, value: float) -> float:
        """Map a data y onto the canvas. SVG y grows downward; data does not."""
        if self.y_hi == self.y_lo:
            return self.y + self.height / 2.0
        t = (value - self.y_lo) / (self.y_hi - self.y_lo)
        return self.y + self.height - min(max(t, 0.0), 1.0) * self.height


def _document(width: float, height: float, body: str, *, title: str) -> str:
    """Wrap a body in a self-describing, theme-neutral SVG document.

    An explicit white plate rather than a transparent background: these files
    are embedded in a report that may be read on a dark background, and a
    transparent chart with dark ink becomes an invisible chart.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
        f'width="{width:g}" height="{height:g}" role="img" '
        f'aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>"
        f'<rect x="0" y="0" width="{width:g}" height="{height:g}" fill="#ffffff"/>'
        f"{body}</svg>"
    )


def _text(
    x: float, y: float, value: str, *, size: float = 11.0, anchor: str = "start", fill: str = _INK
) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" {_FONT} font-size="{size:g}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>'
    )


def _frame(box: Box, *, x_label: str, y_label: str, ticks: int = 5) -> str:
    """Axes, gridlines and tick labels for a linear box."""
    parts = [
        f'<rect x="{box.x:g}" y="{box.y:g}" width="{box.width:g}" height="{box.height:g}" '
        f'fill="none" stroke="{_GRID}"/>'
    ]
    for step in range(ticks + 1):
        t = step / ticks
        xv = box.x_lo + t * (box.x_hi - box.x_lo)
        yv = box.y_lo + t * (box.y_hi - box.y_lo)
        px, py = box.sx(xv), box.sy(yv)
        parts.append(
            f'<line x1="{px:g}" y1="{box.y:g}" x2="{px:g}" y2="{box.y + box.height:g}" '
            f'stroke="{_GRID}" stroke-dasharray="2 3"/>'
        )
        parts.append(
            f'<line x1="{box.x:g}" y1="{py:g}" x2="{box.x + box.width:g}" y2="{py:g}" '
            f'stroke="{_GRID}" stroke-dasharray="2 3"/>'
        )
        parts.append(_text(px, box.y + box.height + 14, f"{xv:.2f}", size=9, anchor="middle"))
        parts.append(_text(box.x - 6, py + 3, f"{yv:.2f}", size=9, anchor="end"))
    parts.append(
        _text(box.x + box.width / 2, box.y + box.height + 30, x_label, size=11, anchor="middle")
    )
    parts.append(
        f'<text x="{box.x - 34:g}" y="{box.y + box.height / 2:g}" {_FONT} font-size="11" '
        f'text-anchor="middle" fill="{_INK}" '
        f'transform="rotate(-90 {box.x - 34:g} {box.y + box.height / 2:g})">'
        f"{escape(y_label)}</text>"
    )
    return "".join(parts)


def reliability_svg(report: CalibrationReport, *, title: str) -> str:
    """§10.5's reliability diagram, with the bin-count histogram underneath.

    The histogram is not decoration and it is not optional: "a reliability
    curve without bin counts hides the fact that the interesting bins may hold
    nine scenarios". Wilson intervals are drawn as vertical whiskers for the
    same reason -- an 8-point gap on 20 scenarios and an 8-point gap on 120 are
    different findings and must not look alike.
    """
    width, height = 560.0, 520.0
    plot = Box(70, 40, 440, 320, 0.0, 1.0, 0.0, 1.0)
    hist = Box(70, 410, 440, 70, 0.0, 1.0, 0.0, 1.0)

    parts = [_text(width / 2, 24, title, size=13, anchor="middle")]
    parts.append(_frame(plot, x_label="mean predicted probability", y_label="observed frequency"))
    parts.append(
        f'<line x1="{plot.sx(0.0):g}" y1="{plot.sy(0.0):g}" x2="{plot.sx(1.0):g}" '
        f'y2="{plot.sy(1.0):g}" stroke="{_MUTED}" stroke-dasharray="5 4"/>'
    )
    parts.append(_text(plot.sx(0.62), plot.sy(0.56), "perfect calibration", size=9, fill=_MUTED))

    points: list[tuple[float, float]] = []
    for row in report.bins:
        if row.count == 0 or row.mean_pred is None or row.obs_freq is None:
            continue
        px, py = plot.sx(row.mean_pred), plot.sy(row.obs_freq)
        points.append((px, py))
        if row.wilson_lo is not None and row.wilson_hi is not None:
            top, bottom = plot.sy(row.wilson_hi), plot.sy(row.wilson_lo)
            parts.append(
                f'<line x1="{px:g}" y1="{top:g}" x2="{px:g}" y2="{bottom:g}" '
                f'stroke="{_ACCENT}" stroke-width="1.2"/>'
                f'<line x1="{px - 3:g}" y1="{top:g}" x2="{px + 3:g}" y2="{top:g}" '
                f'stroke="{_ACCENT}"/>'
                f'<line x1="{px - 3:g}" y1="{bottom:g}" x2="{px + 3:g}" y2="{bottom:g}" '
                f'stroke="{_ACCENT}"/>'
            )
    if len(points) > 1:
        path = " ".join(
            ("M" if index == 0 else "L") + f"{px:g},{py:g}" for index, (px, py) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{_ACCENT}" stroke-width="1.8"/>')
    for px, py in points:
        parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="3.4" fill="{_ACCENT}"/>')

    counts = [row.count for row in report.bins]
    peak = max(counts) if counts else 0
    parts.append(
        f'<rect x="{hist.x:g}" y="{hist.y:g}" width="{hist.width:g}" '
        f'height="{hist.height:g}" fill="none" stroke="{_GRID}"/>'
    )
    bins = max(1, len(report.bins))
    slot = hist.width / bins
    for row in report.bins:
        share = (row.count / peak) if peak else 0.0
        bar = share * hist.height
        x = hist.x + row.index * slot
        parts.append(
            f'<rect x="{x + 1:g}" y="{hist.y + hist.height - bar:g}" '
            f'width="{max(slot - 2, 1):g}" height="{bar:g}" fill="{_MUTED}"/>'
        )
        if row.count:
            parts.append(
                _text(
                    x + slot / 2,
                    hist.y + hist.height - bar - 3,
                    str(row.count),
                    size=8,
                    anchor="middle",
                )
            )
    parts.append(_text(hist.x, hist.y - 6, "scenarios per bin", size=10, fill=_MUTED))
    parts.append(
        _text(
            hist.x + hist.width,
            hist.y + hist.height + 16,
            f"n = {report.n} · ECE {report.ece:.4f} · MCE {report.mce:.4f}",
            size=10,
            anchor="end",
            fill=_MUTED,
        )
    )
    return _document(width, height, "".join(parts), title=title)


def forest_svg(comparisons: Sequence[Comparison], *, title: str) -> str:
    """Effect sizes with their paired-bootstrap intervals (Appendix D).

    Zero is drawn as a solid rule because "the interval crosses zero" is the
    whole content of the chart. An interval whose adjusted p-value survives
    Holm is drawn in full ink and one that does not is drawn muted, so the
    multiple-comparison correction is visible in the figure rather than only
    in a table two pages away.
    """
    rows = list(comparisons)
    width = 620.0
    height = 90.0 + 34.0 * max(len(rows), 1)
    widest = (
        max(max(abs(item.interval.lo), abs(item.interval.hi)) for item in rows) if rows else 0.05
    )
    span = max(widest * 1.15, 1e-3)
    plot = Box(240, 50, 330, 34.0 * max(len(rows), 1), -span, span, 0.0, 1.0)

    parts = [_text(width / 2, 26, title, size=13, anchor="middle")]
    zero = plot.sx(0.0)
    parts.append(
        f'<line x1="{zero:g}" y1="{plot.y:g}" x2="{zero:g}" '
        f'y2="{plot.y + plot.height:g}" stroke="{_INK}" stroke-width="1"/>'
    )
    for index, item in enumerate(rows):
        y = plot.y + 17.0 + index * 34.0
        adjusted = item.p_adjusted
        significant = adjusted is not None and adjusted < 0.05
        colour = _ACCENT if significant else _MUTED
        lo, hi = plot.sx(item.interval.lo), plot.sx(item.interval.hi)
        point = plot.sx(item.interval.point)
        parts.append(
            f'<line x1="{lo:g}" y1="{y:g}" x2="{hi:g}" y2="{y:g}" '
            f'stroke="{colour}" stroke-width="1.6"/>'
            f'<line x1="{lo:g}" y1="{y - 4:g}" x2="{lo:g}" y2="{y + 4:g}" stroke="{colour}"/>'
            f'<line x1="{hi:g}" y1="{y - 4:g}" x2="{hi:g}" y2="{y + 4:g}" stroke="{colour}"/>'
            f'<circle cx="{point:g}" cy="{y:g}" r="3.6" fill="{colour}"/>'
        )
        parts.append(_text(236, y + 4, item.name, size=10, anchor="end"))
        label = f"{item.interval.point:+.4f} [{item.interval.lo:+.4f}, {item.interval.hi:+.4f}]"
        if adjusted is not None:
            label += f"  p*={adjusted:.3g}"
        parts.append(_text(plot.x, plot.y + plot.height + 26, label, size=9, fill=_MUTED))
    for step in (-1.0, -0.5, 0.0, 0.5, 1.0):
        value = step * span
        parts.append(
            _text(
                plot.sx(value), plot.y + plot.height + 14, f"{value:+.3f}", size=9, anchor="middle"
            )
        )
    parts.append(
        _text(
            plot.x + plot.width / 2,
            height - 10,
            "Brier difference (positive = the ablated system is worse)",
            size=10,
            anchor="middle",
            fill=_MUTED,
        )
    )
    return _document(width, height, "".join(parts), title=title)


def convergence_svg(
    points: Sequence[tuple[int, float]], *, title: str, y_label: str = "mean |change in p|"
) -> str:
    """§9.3's convergence curve: how much p_hat still moves as n grows.

    The x axis is the replicate-count ladder by position rather than by value,
    because the ladder doubles and a linear axis would compress every rung
    that matters into the left margin.
    """
    width, height = 540.0, 320.0
    plot = Box(70, 46, 430, 210, 0.0, 1.0, 0.0, 1.0)
    parts = [_text(width / 2, 26, title, size=13, anchor="middle")]

    if not points:
        parts.append(
            _text(
                width / 2, height / 2, "no convergence data", size=12, anchor="middle", fill=_MUTED
            )
        )
        return _document(width, height, "".join(parts), title=title)

    peak = max(value for _, value in points)
    scale = peak if peak > 0 else 1.0
    parts.append(
        f'<rect x="{plot.x:g}" y="{plot.y:g}" width="{plot.width:g}" '
        f'height="{plot.height:g}" fill="none" stroke="{_GRID}"/>'
    )
    positions = [
        plot.x + (index / max(len(points) - 1, 1)) * plot.width for index in range(len(points))
    ]
    path_parts: list[str] = []
    for index, ((n, value), px) in enumerate(zip(points, positions, strict=True)):
        py = plot.y + plot.height - (value / scale) * plot.height
        path_parts.append(("M" if index == 0 else "L") + f"{px:g},{py:g}")
        parts.append(f'<circle cx="{px:g}" cy="{py:g}" r="3.4" fill="{_ACCENT}"/>')
        parts.append(_text(px, plot.y + plot.height + 16, f"n={n}", size=9, anchor="middle"))
        parts.append(_text(px, py - 8, f"{value:.4f}", size=8, anchor="middle", fill=_MUTED))
    parts.append(
        f'<path d="{" ".join(path_parts)}" fill="none" stroke="{_ACCENT}" stroke-width="1.8"/>'
    )
    parts.append(_text(plot.x - 6, plot.y + 4, f"{scale:.4f}", size=9, anchor="end"))
    parts.append(_text(plot.x - 6, plot.y + plot.height + 4, "0", size=9, anchor="end"))
    parts.append(
        _text(plot.x + plot.width / 2, height - 12, y_label, size=10, anchor="middle", fill=_MUTED)
    )
    return _document(width, height, "".join(parts), title=title)


def scatter_svg(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    annotation: str = "",
) -> str:
    """§9.2's "sigma is informative" claim, drawn against the data it rests on.

    The annotation carries the correlation and its p-value. It is passed in
    rather than computed here, so the figure and the significance table can
    never disagree about the same number.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    width, height = 520.0, 400.0
    x_hi = max(xs) if xs else 1.0
    y_hi = max(ys) if ys else 1.0
    plot = Box(70, 46, 400, 280, 0.0, max(x_hi, 1e-6), 0.0, max(y_hi, 1e-6))

    parts = [_text(width / 2, 26, title, size=13, anchor="middle")]
    parts.append(_frame(plot, x_label=x_label, y_label=y_label))
    for x, y in zip(xs, ys, strict=True):
        parts.append(
            f'<circle cx="{plot.sx(x):g}" cy="{plot.sy(y):g}" r="2.8" '
            f'fill="{_ACCENT}" fill-opacity="0.55"/>'
        )
    if annotation:
        parts.append(_text(plot.x + 8, plot.y + 16, annotation, size=10, fill=_WARN))
    return _document(width, height, "".join(parts), title=title)
