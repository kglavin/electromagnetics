"""Module 1 animation: a vector field morphing from pure curl to pure divergence.

Render with:  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py DivergenceVersusCurl

Primer scenes:
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py LineIntegralPaths     (primer 00b, section 3.3)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py InternalFacesCancel   (primer 00d, section 3.4)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py InternalEdgesCancel   (primer 00d, section 3.5)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py DotProductShadow      (primer 00a, section 3.2)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py CrossProductArea      (primer 00a, section 3.3)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py ShrinkingInstruments  (primer 00c, section 3.3)
  uv run manim -ql --media_dir media/m1_fields animations/m1_fields/scenes.py RingConvergence       (primer 00e, section 3.4)
"""

import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    ArrowVectorField,
    Dot,
    Line,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)


def blend_field(s):
    """Field (1 - s) F_rot + s F_exp with F_rot = (-y, x, 0) and F_exp = (x, y, 0).

    Analytically, div = 2 s and curl_z = 2 (1 - s) everywhere, so the paddlewheel spin rate
    and the on-screen numbers follow directly from s.
    """

    def f(p):
        x, y = p[0], p[1]
        return np.array([(1 - s) * (-y) + s * x, (1 - s) * x + s * y, 0.0])

    return f


class DivergenceVersusCurl(Scene):
    def construct(self):
        s = ValueTracker(0.0)

        field = always_redraw(
            lambda: ArrowVectorField(
                blend_field(s.get_value()),
                x_range=[-6, 6, 0.75],
                y_range=[-3, 3, 0.75],
                colors=[BLUE_E, YELLOW],
            )
        )

        # A paddlewheel: two crossed bars on a pivot. Its spin rate is set by curl_z / 2,
        # the local angular velocity of a small rigid object carried by the flow.
        wheel = VGroup(
            Line(0.35 * LEFT, 0.35 * RIGHT, stroke_width=6),
            Line(0.35 * UP, 0.35 * DOWN, stroke_width=6),
            Dot(radius=0.06),
        ).set_color(YELLOW).move_to([2.5, 1.2, 0])
        wheel.add_updater(lambda m, dt: m.rotate((1 - s.get_value()) * dt))

        readout = always_redraw(
            lambda: Text(
                f"div F = {2 * s.get_value():.1f}     curl F = {2 * (1 - s.get_value()):.1f}",
                font_size=32,
            ).to_edge(UP).add_background_rectangle(opacity=0.85, buff=0.15)
        )
        caption = always_redraw(
            lambda: Text(
                "pure rotation: the wheel spins, nothing flows outward"
                if s.get_value() < 0.5
                else "pure expansion: everything flows outward, the wheel stops",
                font_size=26,
            ).to_edge(DOWN).add_background_rectangle(opacity=0.85, buff=0.15)
        )

        self.add(field, wheel, readout, caption)
        self.wait(2.5)
        self.play(s.animate.set_value(1.0), run_time=5, rate_func=linear)
        self.wait(2.5)


# ---------------------------------------------------------------------------
# Primer scenes (00b line integrals, 00d integral theorems)
# ---------------------------------------------------------------------------

from manim import (  # noqa: E402
    GREEN,
    GREY_B,
    ORANGE,
    RED,
    BLUE,
    WHITE,
    Arrow,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Square,
    VMobject,
    Write,
)


def _midpoint_cumulative(F, r_of_t, t0, t1, n=4000):
    """Parameter values and the running midpoint sum of F . dl along r(t)."""
    t = np.linspace(t0, t1, n + 1)
    pts = r_of_t(t)
    dl = np.diff(pts, axis=0)
    mid = 0.5 * (pts[1:] + pts[:-1])
    steps = np.einsum("ij,ij->i", F(mid), dl)
    return t, np.concatenate([[0.0], np.cumsum(steps)])


F_GRAD = lambda p: np.stack([p[..., 1], p[..., 0]], axis=-1)  # (y, x) = grad(xy)  # noqa: E731
F_ROT = lambda p: np.stack([-p[..., 1], p[..., 0]], axis=-1)  # (-y, x)  # noqa: E731
STRAIGHT = lambda t: np.stack([t, t], axis=-1)  # noqa: E731
PARABOLA = lambda t: np.stack([t, t**2], axis=-1)  # noqa: E731
CIRCLE = lambda t: np.stack([np.cos(t), np.sin(t)], axis=-1)  # noqa: E731


class _Panel:
    """Maps data coordinates (x, y) to screen points for one panel."""

    def __init__(self, centre, lo, hi, width):
        self.lo, self.hi = lo, hi
        self.scale = width / (hi - lo)
        self.c = np.array([centre[0], centre[1], 0.0])
        self.mid = 0.5 * (lo + hi)

    def __call__(self, xy):
        x, y = xy[0], xy[1]
        return self.c + self.scale * np.array([x - self.mid, y - self.mid, 0.0])

    def quiver(self, F, step, colour=GREY_B, opacity=0.45, length=0.42):
        pts = np.arange(self.lo + step / 2, self.hi, step)
        X, Y = np.meshgrid(pts, pts, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel()], axis=-1)
        V = F(P)
        vmax = np.linalg.norm(V, axis=1).max()
        arrows = VGroup()
        for p, v in zip(P, V):
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            d = np.array([v[0], v[1], 0.0]) / vmax * length
            s = self(p)
            arrows.add(Arrow(s - d / 2, s + d / 2, buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.3,
                             color=colour, stroke_opacity=opacity, fill_opacity=opacity))
        return arrows

    def curve(self, r_of_t, t0, t1, colour, n=200):
        t = np.linspace(t0, t1, n)
        pts = r_of_t(t)
        return VMobject(stroke_color=colour, stroke_width=4).set_points_as_corners([self(p) for p in pts])


def _field_arrow(panel, F, xy, scale=0.45):
    v = F(np.asarray(xy))
    n = np.linalg.norm(v)
    if n < 1e-3:
        return VMobject()
    s = panel(xy)
    d = np.array([v[0], v[1], 0.0]) * scale
    return Arrow(s, s + d, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.25, color=YELLOW)


class LineIntegralPaths(Scene):
    """Primer M1.0b section 3.3: F . dl summed along two paths with the same endpoints.

    Left F = (y, x) = grad(xy): both paths give 1. Right F = (-y, x): 0 and 1/3.
    Then around the unit circle: 0 for (y, x), 2 pi for (-y, x).
    Every number on screen is the running midpoint sum along the drawn path.
    """

    def construct(self):
        title = Text("Line integrals: does the path matter?", font_size=32).to_edge(UP, buff=0.25)
        self.add(title)

        fields = ((F_GRAD, "F = (y, x)"), (F_ROT, "F = (−y, x)"))
        paths = ((STRAIGHT, "straight line", GREEN), (PARABOLA, "parabola y = x²", ORANGE))
        panels = [_Panel((-3.4, 0.15), -0.2, 1.2, 4.0), _Panel((3.4, 0.15), -0.2, 1.2, 4.0)]

        t = ValueTracker(0.0)
        stage1 = VGroup()
        readouts = VGroup()
        finals = []
        for panel, (F, label) in zip(panels, fields):
            box = Square(side_length=4.0, stroke_color=GREY_B, stroke_width=1).move_to(panel.c)
            name = Text(label, font_size=26).next_to(box, UP, buff=0.12)
            stage1.add(box, name, panel.quiver(F, 0.2))
            row = []
            for k, (r, pname, colour) in enumerate(paths):
                tt, cum = _midpoint_cumulative(F, r, 0.0, 1.0)
                finals.append(cum[-1])
                stage1.add(panel.curve(r, 0.0, 1.0, colour))
                dot = Dot(radius=0.08, color=colour)
                dot.add_updater(lambda m, r=r, panel=panel: m.move_to(panel(r(t.get_value()))))
                arrow = always_redraw(lambda r=r, panel=panel, F=F: _field_arrow(panel, F, r(t.get_value())))
                text = always_redraw(
                    lambda tt=tt, cum=cum, pname=pname, colour=colour, box=box, k=k: Text(
                        f"{pname}:  ∫F·dl = {np.interp(t.get_value(), tt, cum):6.3f}", font_size=22, color=colour
                    ).next_to(box, DOWN, buff=0.15 + 0.42 * k)
                )
                row += [arrow, dot, text]
            readouts.add(*row)
        start_dots = VGroup(*[Dot(p((0.0, 0.0)), radius=0.06) for p in panels], *[Dot(p((1.0, 1.0)), radius=0.06) for p in panels])

        self.play(FadeIn(stage1), FadeIn(start_dots), run_time=1.5)
        self.add(readouts)
        self.wait(1.0)
        self.play(t.animate.set_value(1.0), run_time=6, rate_func=linear)
        self.wait(0.5)
        verdicts = VGroup(
            Text("same endpoints, same value: path independent", font_size=20).next_to(panels[0].c, DOWN, buff=3.15),
            Text("same endpoints, different values: path dependent", font_size=20).next_to(panels[1].c, DOWN, buff=3.15),
        )
        self.play(FadeIn(verdicts))
        self.wait(2.5)

        # Closed loop: the unit circle, counter-clockwise.
        for m in readouts:
            m.clear_updaters()
        self.play(FadeOut(stage1), FadeOut(readouts), FadeOut(verdicts), FadeOut(start_dots))
        panels2 = [_Panel((-3.4, 0.15), -1.35, 1.35, 4.0), _Panel((3.4, 0.15), -1.35, 1.35, 4.0)]
        s = ValueTracker(0.0)
        stage2 = VGroup()
        loop_readouts = VGroup()
        for panel, (F, label) in zip(panels2, fields):
            box = Square(side_length=4.0, stroke_color=GREY_B, stroke_width=1).move_to(panel.c)
            name = Text(label + "   around the unit circle", font_size=24).next_to(box, UP, buff=0.12)
            tt, cum = _midpoint_cumulative(F, CIRCLE, 0.0, 2 * np.pi)
            finals.append(cum[-1])
            stage2.add(box, name, panel.quiver(F, 0.3), Circle(radius=panel.scale, color=WHITE, stroke_width=3).move_to(panel.c))
            dot = Dot(radius=0.08, color=WHITE)
            dot.add_updater(lambda m, panel=panel: m.move_to(panel(CIRCLE(s.get_value()))))
            arrow = always_redraw(lambda panel=panel, F=F: _field_arrow(panel, F, CIRCLE(s.get_value()), scale=0.6))
            text = always_redraw(
                lambda tt=tt, cum=cum, box=box: Text(
                    f"∮F·dl so far = {np.interp(s.get_value(), tt, cum):6.3f}", font_size=24
                ).next_to(box, DOWN, buff=0.2)
            )
            loop_readouts.add(arrow, dot, text)
        self.play(FadeIn(stage2), run_time=1.2)
        self.add(loop_readouts)
        self.play(s.animate.set_value(2 * np.pi), run_time=7, rate_func=linear)
        verdicts2 = VGroup(
            Text("closed loop gives 0: a potential exists", font_size=20).next_to(panels2[0].c, DOWN, buff=2.75),
            Text("closed loop gives 2π = 6.283: no potential", font_size=20).next_to(panels2[1].c, DOWN, buff=2.75),
        )
        self.play(FadeIn(verdicts2))
        self.wait(3)


def _cells(lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    return [((edges[i], edges[j]), (edges[i + 1], edges[j + 1])) for i in range(n) for j in range(n)]


_GX, _GW = np.polynomial.legendre.leggauss(8)


def _edge_integral(f, a, b):
    """Gauss-Legendre integral of a scalar function of the parameter s in [a, b] (exact for these polynomials)."""
    s = 0.5 * (b - a) * _GX + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(_GW * f(s))


def _cell_faces(F, cell, mode):
    """Per-side contributions of one cell.

    Returns a list of (midpoint, normal, tangent, value). For mode "flux" value is the outward flux
    through the side; for mode "circ" it is the counter-clockwise line integral along the side.
    normal is the outward normal, tangent the counter-clockwise direction along the side.
    """
    (x0, y0), (x1, y1) = cell
    out = []
    sides = (
        ((x1, None), np.array([1.0, 0.0]), np.array([0.0, 1.0])),   # right
        ((None, y1), np.array([0.0, 1.0]), np.array([-1.0, 0.0])),  # top
        ((x0, None), np.array([-1.0, 0.0]), np.array([0.0, -1.0])),  # left
        ((None, y0), np.array([0.0, -1.0]), np.array([1.0, 0.0])),  # bottom
    )
    for (xf, yf), n, tdir in sides:
        if xf is not None:
            g = lambda s, xf=xf: F(np.stack([np.full_like(s, xf), s], axis=-1))  # noqa: E731
            vec = lambda s, g=g: g(s)  # noqa: E731
            a, b = y0, y1
            mid = np.array([xf, 0.5 * (y0 + y1)])
        else:
            g = lambda s, yf=yf: F(np.stack([s, np.full_like(s, yf)], axis=-1))  # noqa: E731
            vec = lambda s, g=g: g(s)  # noqa: E731
            a, b = x0, x1
            mid = np.array([0.5 * (x0 + x1), yf])
        w = n if mode == "flux" else tdir
        value = _edge_integral(lambda s, vec=vec, w=w: vec(s) @ w, a, b)
        out.append((mid, n, tdir, value))
    return out


def _side_key(mid):
    return (round(float(mid[0]), 9), round(float(mid[1]), 9))


class _Cancellation(Scene):
    """Shared construction for the divergence-theorem and Stokes scenes."""

    MODE = "flux"
    FIELD = None
    LO, HI = 0.25, 1.25
    TITLE = ""
    LEGEND = ""
    CELL_LABEL = ""
    SUM_LABEL = ""
    RIM_LABEL = ""
    INTERNAL_LABEL = ""
    CLOSING = ""

    def arrows_for(self, panel, cells, scale_max):
        """Arrows for every side of every cell, keyed by side so internal pairs can be found."""
        h = (self.HI - self.LO) / int(round(np.sqrt(len(cells))))
        L = min(0.36 * h * panel.scale, 0.9)
        by_side = {}
        group = VGroup()
        for cell in cells:
            for mid, n, tdir, value in _cell_faces(self.FIELD, cell, self.MODE):
                if abs(value) < 1e-12:
                    continue
                length = L * np.sqrt(abs(value) / scale_max)
                length = max(length, 0.12)
                colour = RED if value > 0 else BLUE
                if self.MODE == "flux":
                    # Along the actual flow through the side; red leaves this cell, blue enters it.
                    d = np.sign(value) * n
                    offset = 0.07 * np.array([-n[1], n[0]])
                    centre = panel(mid) + np.array([offset[0], offset[1], 0.0])
                else:
                    # Along the direction F pushes along the side; red helps this cell's
                    # counter-clockwise circulation, blue opposes it. Drawn just inside the cell.
                    d = np.sign(value) * tdir
                    centre = panel(mid) - 0.11 * np.array([n[0], n[1], 0.0])
                d3 = np.array([d[0], d[1], 0.0]) * length
                arrow = Arrow(centre - d3 / 2, centre + d3 / 2, buff=0, stroke_width=4,
                              max_tip_length_to_length_ratio=0.3, color=colour)
                group.add(arrow)
                by_side.setdefault(_side_key(mid), []).append(arrow)
        return group, by_side

    def cell_numbers(self, panel, cells, font_size):
        group = VGroup()
        for cell in cells:
            total = sum(v for _, _, _, v in _cell_faces(self.FIELD, cell, self.MODE))
            lo, hi = np.array(cell[0]), np.array(cell[1])
            centre = panel(lo + np.array([0.55, 0.72]) * (hi - lo))  # clear of the side arrows
            group.add(Text(f"{total:.3f}", font_size=font_size, color=WHITE).move_to(centre))
        return group

    def stage_sums(self, cells):
        total = sum(sum(v for _, _, _, v in _cell_faces(self.FIELD, c, self.MODE)) for c in cells)
        return total

    def construct(self):
        title = Text(self.TITLE, font_size=30).to_edge(UP, buff=0.25)
        legend = Text(self.LEGEND, font_size=20, color=GREY_B).to_edge(DOWN, buff=0.25)
        panel = _Panel((-2.7, -0.15), self.LO, self.HI, 5.0)
        square = Square(side_length=5.0, stroke_color=WHITE, stroke_width=3).move_to(panel.c)
        quiver = panel.quiver(self.FIELD, 0.2, colour=YELLOW, opacity=0.22, length=0.5)
        rim = self.stage_sums(_cells(self.LO, self.HI, 1))

        n_cells = ValueTracker(1)
        n_internal = ValueTracker(0)
        sum_cells = ValueTracker(rim)
        x_text = 1.2

        def line(k, txt, colour=WHITE):
            return Text(txt, font_size=24, color=colour).move_to([x_text, 1.6 - 0.6 * k, 0], aligned_edge=LEFT)

        info = VGroup(
            always_redraw(lambda: line(0, f"cells: {int(round(n_cells.get_value()))}")),
            always_redraw(lambda: line(1, f"{self.SUM_LABEL} = {sum_cells.get_value():.3f}")),
            line(2, f"{self.RIM_LABEL} = {rim:.3f}"),
            always_redraw(lambda: line(3, f"{self.INTERNAL_LABEL}: {int(round(n_internal.get_value()))}", GREY_B)),
        )

        self.add(title, legend)
        self.play(Create(square), FadeIn(quiver), run_time=1.2)
        arrows, _ = self.arrows_for(panel, _cells(self.LO, self.HI, 1), self.max_side(1))
        self.play(FadeIn(arrows), FadeIn(info), run_time=1.2)
        self.wait(2.0)

        current_arrows, current_grid, current_numbers = arrows, VGroup(), VGroup()
        for n, font_size, cancel_time in ((2, 26, 0.9), (4, 16, 4.0)):
            cells = _cells(self.LO, self.HI, n)
            edges = np.linspace(self.LO, self.HI, n + 1)[1:-1]
            grid = VGroup(*[Line(panel((e, self.LO)), panel((e, self.HI)), stroke_width=2, color=GREY_B) for e in edges],
                          *[Line(panel((self.LO, e)), panel((self.HI, e)), stroke_width=2, color=GREY_B) for e in edges])
            new_arrows, by_side = self.arrows_for(panel, cells, self.max_side(n))
            numbers = self.cell_numbers(panel, cells, font_size)
            pairs = [v for v in by_side.values() if len(v) == 2]
            fade = [FadeOut(current_arrows), FadeOut(current_numbers), FadeOut(current_grid)]
            if n == 2:
                fade.append(FadeOut(quiver))
            self.play(*fade, run_time=0.6)
            self.play(Create(grid), run_time=0.8)
            n_cells.set_value(n * n)
            sum_cells.set_value(self.stage_sums(cells))
            n_internal.set_value(len(pairs))
            self.play(FadeIn(new_arrows), FadeIn(numbers), run_time=1.0)
            self.wait(2.0)
            if n == 2:
                for pair in pairs:
                    self.play(Indicate(VGroup(*pair), scale_factor=1.3, color=WHITE), run_time=0.6)
                    self.play(FadeOut(VGroup(*pair)), n_internal.animate.increment_value(-1), run_time=0.4)
            else:
                self.play(LaggedStart(*[FadeOut(VGroup(*p)) for p in pairs], lag_ratio=0.08),
                          n_internal.animate.set_value(0), run_time=cancel_time)
            for p in pairs:
                for a in p:
                    new_arrows.remove(a)
            self.wait(1.5)
            current_arrows, current_grid, current_numbers = new_arrows, grid, numbers

        closing = Text(self.CLOSING, font_size=24, color=YELLOW).move_to([x_text, -1.2, 0], aligned_edge=LEFT)
        self.play(Write(closing), run_time=1.2)
        self.wait(3)

    def max_side(self, n):
        cells = _cells(self.LO, self.HI, n)
        return max(abs(v) for c in cells for _, _, _, v in _cell_faces(self.FIELD, c, self.MODE))


class InternalFacesCancel(_Cancellation):
    """Primer M1.0d section 3.4: the divergence theorem as cancellation of internal faces.

    F = (x², xy) on the square [0.25, 1.25]². Outward flux of every cell side by Gauss-Legendre
    quadrature. Every internal face appears twice with opposite signs, so the sum over cells equals
    the flux through the boundary, 2.250, which is also the integral of div F = 3x over the square.
    """

    MODE = "flux"
    FIELD = staticmethod(lambda p: np.stack([p[..., 0] ** 2, p[..., 0] * p[..., 1]], axis=-1))
    TITLE = "Divergence theorem: internal faces cancel"
    LEGEND = "F = (x², xy).  Arrows show flow through each side: red leaves that cell, blue enters it."
    SUM_LABEL = "sum of cell fluxes"
    RIM_LABEL = "flux out of the boundary"
    INTERNAL_LABEL = "internal faces left"
    CLOSING = "∫ ∇·F dA = sum over cells = ∮ F·n dl"


class InternalEdgesCancel(_Cancellation):
    """Primer M1.0d section 3.5: Stokes' theorem as cancellation of internal edges.

    F = (−y, x) on the square [0.25, 1.25]². Counter-clockwise line integral along every cell side.
    Neighbouring cells traverse a shared edge in opposite directions, so the sum of cell circulations
    equals the circulation around the rim, 2.000 = curl F (= 2) times the area.
    """

    MODE = "circ"
    FIELD = staticmethod(lambda p: np.stack([-p[..., 1], p[..., 0]], axis=-1))
    TITLE = "Stokes' theorem: internal edges cancel"
    LEGEND = "F = (−y, x).  Red pushes that cell's loop counter-clockwise, blue pushes against it."
    SUM_LABEL = "sum of cell circulations"
    RIM_LABEL = "circulation around the rim"
    INTERNAL_LABEL = "internal edges left"
    CLOSING = "∫ (∇×F)·n dA = sum over cells = ∮ F·dl"


# ---------------------------------------------------------------------------
# Primer scenes, round two (00a dot and cross product, 00c shrinking instruments, 00e ring)
# ---------------------------------------------------------------------------

from manim import DashedLine, Polygon, TracedPath  # noqa: E402,F401

from emlib import fields as _fields  # noqa: E402

A_VEC = np.array([3.0, 1.0])  # worked examples 2 and 4 in primer 00a
B_LEN = np.sqrt(5.0)
B_START = np.arctan2(2.0, 1.0)  # b = (1, 2)
A_ANGLE = np.arctan2(1.0, 3.0)


class _TwoVectors(Scene):
    """Shared geometry: fixed a, and b of fixed length rotating about the origin."""

    origin = np.array([-4.0, -0.6, 0.0])
    unit = 0.85

    def p(self, v):
        return self.origin + self.unit * np.array([v[0], v[1], 0.0])

    @staticmethod
    def b_of(phi):
        return B_LEN * np.array([np.cos(phi), np.sin(phi)])

    def arrows(self, phi):
        a = Arrow(self.p((0, 0)), self.p(A_VEC), buff=0, color=BLUE, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        a_lab = Text("a = (3, 1)", font_size=24, color=BLUE).next_to(self.p(A_VEC), RIGHT, buff=0.15)
        b = always_redraw(
            lambda: Arrow(self.p((0, 0)), self.p(self.b_of(phi.get_value())), buff=0, color=ORANGE, stroke_width=6,
                          max_tip_length_to_length_ratio=0.15)
        )
        b_lab = always_redraw(
            lambda: Text("b", font_size=26, color=ORANGE).move_to(self.p(1.18 * self.b_of(phi.get_value())))
        )
        return VGroup(a, a_lab), VGroup(b, b_lab)

    def sweep(self, phi, stops, run=3.0, hold=1.4):
        """Rotate b through the listed angles (radians), pausing at each."""
        for target in stops:
            self.play(phi.animate.set_value(target), run_time=run, rate_func=linear)
            self.wait(hold)


class DotProductShadow(_TwoVectors):
    """Primer 00a, section 3.2: a . b as the shadow of a on the line of b, as b turns a full circle."""

    def construct(self):
        phi = ValueTracker(B_START)
        title = Text("The dot product: the shadow of a on the line of b", font_size=32).to_edge(UP, buff=0.3)
        a_grp, b_grp = self.arrows(phi)

        def geom():
            bh = self.b_of(phi.get_value()) / B_LEN
            s = A_VEC @ bh
            foot = s * bh
            line = DashedLine(self.p(-3.6 * bh), self.p(3.6 * bh), color=GREY_B, stroke_width=2, dash_length=0.12)
            colour = GREEN if s > 0.02 else (RED if s < -0.02 else WHITE)
            shadow = Line(self.p((0, 0)), self.p(foot), color=colour, stroke_width=10)
            drop = DashedLine(self.p(A_VEC), self.p(foot), color=WHITE, stroke_width=2, dash_length=0.08)
            return VGroup(line, shadow, drop)

        g = always_redraw(geom)

        def readout():
            b = self.b_of(phi.get_value())
            d = A_VEC @ b
            th = np.degrees(np.arccos(np.clip(d / (np.linalg.norm(A_VEC) * B_LEN), -1, 1)))
            s = d / B_LEN
            if s > 0.02:
                note, col = "below 90°: shadow along b, a · b > 0", GREEN
            elif s < -0.02:
                note, col = "above 90°: shadow against b, a · b < 0", RED
            else:
                note, col = "perpendicular: no shadow, a · b = 0", WHITE
            lines = VGroup(
                Text(f"b = ({b[0]:+.2f}, {b[1]:+.2f})", font_size=26, color=ORANGE),
                Text(f"angle between a and b = {th:5.1f}°", font_size=26),
                Text(f"a · b = ax bx + ay by = {d:+.2f}", font_size=26),
                Text(f"shadow length a · b/|b| = {s:+.2f}", font_size=26, color=col),
                Text(note, font_size=22, color=col),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
            return lines.move_to([3.4, 0.4, 0]).align_to(np.array([0.7, 0, 0]), LEFT)

        r = always_redraw(readout)
        caption = Text("a · b = |a| |b| cos θ   (worked example 2: a = (3, 1), b = (1, 2) gives 5 at 45°)",
                       font_size=20, color=GREY_B).to_edge(DOWN, buff=0.3)

        self.play(Write(title))
        self.play(FadeIn(a_grp), FadeIn(b_grp), FadeIn(g), FadeIn(r), FadeIn(caption))
        self.wait(2.5)
        self.sweep(phi, [A_ANGLE + np.pi / 2, A_ANGLE + np.pi, A_ANGLE + 1.5 * np.pi, B_START + 2 * np.pi])
        self.wait(1.0)


class CrossProductArea(_TwoVectors):
    """Primer 00a, section 3.3: |a x b| as the area of the parallelogram, and the sign of (a x b)_z."""

    def construct(self):
        phi = ValueTracker(B_START)
        title = Text("The cross product: area of the parallelogram, direction by the right hand", font_size=28).to_edge(UP, buff=0.3)
        a_grp, b_grp = self.arrows(phi)

        def para():
            b = self.b_of(phi.get_value())
            z = A_VEC[0] * b[1] - A_VEC[1] * b[0]
            col = BLUE if z > 0.02 else (ORANGE if z < -0.02 else GREY_B)
            poly = Polygon(self.p((0, 0)), self.p(A_VEC), self.p(A_VEC + b), self.p(b), stroke_width=1,
                           stroke_color=col, fill_color=col, fill_opacity=0.35)
            sym = "⊙" if z > 0.02 else ("⊗" if z < -0.02 else "")
            mark = Text(sym, font_size=40, color=WHITE).move_to(self.p(0.5 * (A_VEC + b)))
            return VGroup(poly, mark)

        g = always_redraw(para)

        def readout():
            b = self.b_of(phi.get_value())
            z = A_VEC[0] * b[1] - A_VEC[1] * b[0]
            turn = np.degrees((phi.get_value() - A_ANGLE) % (2 * np.pi))
            if z > 0.02:
                note, col = "b counter-clockwise from a: out of the page ⊙", BLUE
            elif z < -0.02:
                note, col = "b clockwise from a: into the page ⊗", ORANGE
            else:
                note, col = "a and b parallel: no area, a × b = 0", WHITE
            lines = VGroup(
                Text(f"b = ({b[0]:+.2f}, {b[1]:+.2f})", font_size=26, color=ORANGE),
                Text(f"turn a to b, counter-clockwise: {turn:5.1f}°", font_size=22),
                Text(f"area = |a| |b| |sin θ| = {abs(z):.2f}", font_size=26),
                Text(f"(a × b)z = ax by - ay bx = {z:+.2f}", font_size=26, color=col),
                Text(note, font_size=22, color=col),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
            return lines.move_to([3.6, 0.4, 0]).align_to(np.array([0.9, 0, 0]), LEFT)

        r = always_redraw(readout)
        caption = Text("worked example 4: a = (3, 1), b = (1, 2) gives a × b = (0, 0, 5); b × a = -(a × b)",
                       font_size=20, color=GREY_B).to_edge(DOWN, buff=0.3)

        self.play(Write(title))
        self.play(FadeIn(a_grp), FadeIn(g), FadeIn(b_grp), FadeIn(r), FadeIn(caption))
        self.wait(2.5)
        self.sweep(phi, [A_ANGLE + np.pi / 2, A_ANGLE + np.pi, A_ANGLE + 1.5 * np.pi, B_START + 2 * np.pi])
        self.wait(1.0)


# Shrinking instruments (primer 00c). Quadratic fields give the exact divergence or curl at every box
# size, because the symmetric box cancels the linear error terms; the cubic fields here leave an h^2 error
# so the convergence is visible.


def _flux_out_of_square(Fx, Fy, cx, cy, half, n=200):
    s = cx + np.linspace(-half, half, n)
    t = cy + np.linspace(-half, half, n)
    ds = s[1] - s[0]
    return (np.trapezoid(Fx(np.full_like(t, cx + half), t), dx=ds) - np.trapezoid(Fx(np.full_like(t, cx - half), t), dx=ds)
            + np.trapezoid(Fy(s, np.full_like(s, cy + half)), dx=ds) - np.trapezoid(Fy(s, np.full_like(s, cy - half)), dx=ds))


def _circulation_around_square(Fx, Fy, cx, cy, half, n=200):
    s = np.linspace(-half, half, n)
    ds = s[1] - s[0]
    return (np.trapezoid(Fx(cx + s, np.full_like(s, cy - half)), dx=ds) + np.trapezoid(Fy(np.full_like(s, cx + half), cy + s), dx=ds)
            - np.trapezoid(Fx(cx + s, np.full_like(s, cy + half)), dx=ds) - np.trapezoid(Fy(np.full_like(s, cx - half), cy + s), dx=ds))


DIV_FIELD = (lambda x, y: x**3, lambda x, y: x * y**2)  # div = 3x^2 + 2xy  # noqa: E731
DIV_POINT, DIV_EXACT = (0.5, 0.4), 3 * 0.5**2 + 2 * 0.5 * 0.4  # 1.15
CURL_FIELD = (lambda x, y: -(y**3), lambda x, y: x**3)  # curl_z = 3x^2 + 3y^2  # noqa: E731
CURL_POINT, CURL_EXACT = (0.3, -0.4), 3 * 0.3**2 + 3 * 0.4**2  # 0.75


class _Instrument:
    """One panel: a map with a shrinking square, and a convergence plot underneath."""

    def __init__(self, x0, field, point, exact, mode, y_range):
        self.Fx, self.Fy = field
        self.c = np.array(point)
        self.exact, self.mode = exact, mode
        self.map_c = np.array([x0, 1.05, 0.0])
        self.scale = 2.6 / 2.5  # data window c +- 1.25 onto 2.6 screen units
        self.plot_lo = np.array([x0 - 2.3, -3.35, 0.0])
        self.plot_w, self.plot_h = 4.6, 1.6
        self.y_range = y_range
        self.ref = max(abs(v[2]) for v in self._side_values(1.0))

    def _side_values(self, h):
        """Twelve samples on the square's sides: (point, direction, F . direction)."""
        out = []
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for f in (-0.6, 0.0, 0.6):
                if sx:
                    pt = self.c + np.array([sx * h, f * h])
                    tang = np.array([0.0, float(sx)])  # counter-clockwise direction on the vertical sides
                else:
                    pt = self.c + np.array([f * h, sy * h])
                    tang = np.array([-float(sy), 0.0])
                F = np.array([self.Fx(*pt), self.Fy(*pt)])
                dirv = np.array([sx, sy], dtype=float) if self.mode == "div" else tang
                out.append((pt, dirv, F @ dirv))
        return out

    def m(self, xy):
        d = np.asarray(xy, dtype=float) - self.c
        return self.map_c + self.scale * np.array([d[0], d[1], 0.0])

    def value(self, h):
        f = _flux_out_of_square if self.mode == "div" else _circulation_around_square
        return f(self.Fx, self.Fy, self.c[0], self.c[1], h) / (2 * h) ** 2

    def q(self, u, v):
        """Plot coordinates: u = log10(1/h) in [0, 2], v the measured value."""
        lo, hi = self.y_range
        return self.plot_lo + np.array([u / 2 * self.plot_w, (v - lo) / (hi - lo) * self.plot_h, 0.0])

    def background(self):
        g = VGroup()
        frame = Square(side_length=2.6, color=GREY_B, stroke_width=1).move_to(self.map_c)
        g.add(frame)
        pts = np.arange(-1.125, 1.126, 0.25)
        vals = [(self.Fx(self.c[0] + a, self.c[1] + b), self.Fy(self.c[0] + a, self.c[1] + b)) for a in pts for b in pts]
        vmax = max(np.hypot(*v) for v in vals)
        k = 0
        for a in pts:
            for b in pts:
                raw = np.array([*vals[k], 0.0])
                v = raw / (np.linalg.norm(raw) + 1e-12) * 0.24 * (np.linalg.norm(raw) / vmax) ** 0.5
                k += 1
                s = self.m(self.c + [a, b])
                if np.linalg.norm(v) > 1e-3:
                    g.add(Arrow(s - v / 2, s + v / 2, buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.35,
                                color=GREY_B, stroke_opacity=0.5, fill_opacity=0.5))
        g.add(Dot(self.m(self.c), radius=0.04, color=YELLOW))
        # convergence plot axes
        lo, hi = self.y_range
        g.add(Line(self.q(0, lo), self.q(2, lo), color=GREY_B, stroke_width=2))
        g.add(Line(self.q(0, lo), self.q(0, hi), color=GREY_B, stroke_width=2))
        g.add(DashedLine(self.q(0, self.exact), self.q(2, self.exact), color=WHITE, stroke_width=2, dash_length=0.08))
        g.add(Text(f"exact {self.exact:.4f}", font_size=16).next_to(self.q(2, self.exact), UP, buff=0.05).shift(0.4 * LEFT))
        for u, lab in ((0, "h = 1"), (1, "0.1"), (2, "0.01")):
            g.add(Text(lab, font_size=16, color=GREY_B).next_to(self.q(u, lo), DOWN, buff=0.08))
        return g

    def live(self, tracker):
        def build():
            u = tracker.get_value()
            h = 10 ** (-u)
            g = VGroup()
            side = 2 * h * self.scale
            g.add(Square(side_length=side, color=WHITE, stroke_width=3).move_to(self.m(self.c)))
            if h > 0.12:  # side arrows, while there is room to see them
                fade = min(1.0, (h - 0.12) / 0.2)
                for pt, dirv, val in self._side_values(h):
                    L = 0.5 * val / self.ref * min(1.0, h / 0.5)
                    if abs(L) < 0.03:
                        continue
                    base = self.m(pt)
                    col = RED if val > 0 else BLUE
                    g.add(Arrow(base, base + L * np.array([dirv[0], dirv[1], 0.0]), buff=0, stroke_width=3,
                                max_tip_length_to_length_ratio=0.3, color=col, stroke_opacity=fade, fill_opacity=fade))
            us = np.linspace(0, u, max(2, int(60 * u) + 2))
            curve = VMobject(stroke_color=YELLOW, stroke_width=3).set_points_as_corners(
                [self.q(x, np.clip(self.value(10 ** (-x)), *self.y_range)) for x in us])
            g.add(curve, Dot(self.q(u, np.clip(self.value(h), *self.y_range)), radius=0.05, color=YELLOW))
            name = "flux / area" if self.mode == "div" else "circulation / area"
            g.add(Text(f"half-width h = {h:.3f}", font_size=20).move_to(self.map_c + np.array([0, -1.75, 0])))
            g.add(Text(f"{name} = {self.value(h):.4f}", font_size=22, color=YELLOW).move_to(self.map_c + np.array([0, -2.1, 0])))
            return g

        return always_redraw(build)


class ShrinkingInstruments(Scene):
    """Primer 00c, sections 3.3 and 3.4: divergence and curl as the limits of flux and circulation per area."""

    def construct(self):
        title = Text("Shrink the instrument: the ratio converges to the derivative", font_size=30).to_edge(UP, buff=0.25)
        left = _Instrument(-3.5, DIV_FIELD, DIV_POINT, DIV_EXACT, "div", (1.0, 2.3))
        right = _Instrument(3.5, CURL_FIELD, CURL_POINT, CURL_EXACT, "curl", (0.6, 2.9))
        heads = VGroup(
            Text("divergence: flux out of a box", font_size=24).move_to([-3.5, 3.05, 0]),
            Text("F = (x³, xy²) at (0.5, 0.4),  ∇·F = 3x² + 2xy", font_size=18, color=GREY_B).move_to([-3.5, 2.65, 0]),
            Text("curl: circulation around a loop", font_size=24).move_to([3.5, 3.05, 0]),
            Text("F = (-y³, x³) at (0.3, -0.4),  (∇×F)z = 3x² + 3y²", font_size=18, color=GREY_B).move_to([3.5, 2.65, 0]),
        )
        u = ValueTracker(0.0)
        caption = Text("red: field leaving the box or running with the loop; blue: entering or against. The error falls as h².",
                       font_size=17, color=GREY_B).to_edge(DOWN, buff=0.12)
        self.play(Write(title), FadeIn(heads))
        self.play(FadeIn(left.background()), FadeIn(right.background()))
        self.add(left.live(u), right.live(u))
        self.play(FadeIn(caption))
        self.wait(2.0)
        self.play(u.animate.set_value(1.0), run_time=5, rate_func=linear)
        self.wait(1.5)
        self.play(u.animate.set_value(2.0), run_time=4, rate_func=linear)
        self.wait(2.5)


# Ring of point charges (primer 00e, section 3.4). Units: k Q / R^2, with R = 1.

RING_NS = (4, 8, 16, 32, 64)
P_FIELD = np.array([[1.5, 0.0, 0.0]])  # in the ring's plane, between two charges


def _ring(N):
    phi = 2 * np.pi * (np.arange(N) + 0.5) / N
    return np.stack([np.cos(phi), np.sin(phi), 0 * phi], axis=-1), np.full(N, 1.0 / N)


def _ring_E(N, obs):
    src, q = _ring(N)
    return _fields.point_charges_E(src, q, obs) / _fields.K


_E_P_REF = _ring_E(20000, P_FIELD)[0, 0]
_Z_AXIS = np.linspace(0.0, 3.0, 13)
_AXIS_EXACT = _Z_AXIS / (1 + _Z_AXIS**2) ** 1.5
RING_ROWS = []
for _N in RING_NS:
    _Ez = _ring_E(_N, np.stack([0 * _Z_AXIS, 0 * _Z_AXIS, _Z_AXIS], axis=-1))[:, 2]
    _EP = _ring_E(_N, P_FIELD)[0, 0]
    RING_ROWS.append((_N, _Ez, np.abs(_Ez - _AXIS_EXACT).max(), _EP, abs(_EP / _E_P_REF - 1)))


class RingConvergence(Scene):
    """Primer 00e, section 3.4: a ring as N point charges. Exact on the axis for any N, slow near the ring."""

    def construct(self):
        title = Text("A continuous ring as N point charges", font_size=32).to_edge(UP, buff=0.3)
        C, Rs = np.array([-4.2, 0.0, 0.0]), 1.5
        ring = Circle(radius=Rs, color=GREY_B, stroke_width=2).move_to(C)
        axis_mark = VGroup(Text("⊙", font_size=30), Text("axis out of the page", font_size=16, color=GREY_B).move_to(C + [0, -0.4, 0]))
        axis_mark[0].move_to(C)
        P = C + np.array([1.5 * Rs, 0, 0])
        p_mark = VGroup(Dot(P, radius=0.07, color=YELLOW), Text("P, r = 1.5R", font_size=18, color=YELLOW).next_to(P, UP, buff=0.12))
        ring_lab = Text("radius R, total charge Q", font_size=18, color=GREY_B).move_to(C + [0, -2.1, 0])

        # axis plot, top right
        o1, w1, h1 = np.array([0.6, 0.4, 0]), 4.2, 2.2
        ax_q = lambda z, e: o1 + np.array([z / 3.0 * w1, e / 0.45 * h1, 0])  # noqa: E731
        zz = np.linspace(0, 3, 200)
        axis_plot = VGroup(
            Line(ax_q(0, 0), ax_q(3, 0), color=GREY_B, stroke_width=2), Line(ax_q(0, 0), ax_q(0, 0.45), color=GREY_B, stroke_width=2),
            VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners([ax_q(z, z / (1 + z**2) ** 1.5) for z in zz]),
            Text("on the axis: Ez in units of kQ/R²", font_size=18).next_to(ax_q(1.5, 0.45), UP, buff=0.1),
            Text("z/R: 0", font_size=14, color=GREY_B).next_to(ax_q(0, 0), DOWN, buff=0.08),
            Text("3", font_size=14, color=GREY_B).next_to(ax_q(3, 0), DOWN, buff=0.08),
            Text("white: kQz/(R² + z²)^1.5", font_size=14, color=GREY_B).move_to(ax_q(2.2, 0.38)),
        )

        # error plot at P, bottom right: log10(relative error) against N
        o2, w2, h2 = np.array([0.6, -3.3, 0]), 4.2, 2.1
        er_q = lambda i, le: o2 + np.array([(i + 0.5) / len(RING_NS) * w2, (le + 12) / 12 * h2, 0])  # noqa: E731
        err_plot = VGroup(
            Line(o2, o2 + [w2, 0, 0], color=GREY_B, stroke_width=2), Line(o2, o2 + [0, h2, 0], color=GREY_B, stroke_width=2),
            Text("at P: relative error (log scale)", font_size=18).next_to(o2 + [w2 / 2, h2, 0], UP, buff=0.1),
        )
        for le in (0, -4, -8, -12):
            err_plot.add(Text(f"1e{le}" if le else "1", font_size=13, color=GREY_B).next_to(er_q(-0.5, le), LEFT, buff=0.05))
        for i, N in enumerate(RING_NS):
            err_plot.add(Text(str(N), font_size=14, color=GREY_B).next_to(er_q(i, -12), DOWN, buff=0.08))

        self.play(Write(title))
        self.play(Create(ring), FadeIn(axis_mark), FadeIn(p_mark), FadeIn(ring_lab), FadeIn(axis_plot), FadeIn(err_plot))

        charges = VGroup()
        axis_dots = VGroup()
        readout = VGroup()
        err_pts = VGroup()
        for i, (N, Ez, ax_err, EP, rel) in enumerate(RING_ROWS):
            src, _ = _ring(N)
            new_charges = VGroup(*[Dot(C + Rs * np.array([s[0], s[1], 0]), radius=0.07 if N <= 16 else 0.045, color=RED) for s in src])
            new_axis = VGroup(*[Dot(ax_q(z, e), radius=0.05, color=ORANGE) for z, e in zip(_Z_AXIS, Ez)])
            ax_txt = "< 1e-12" if ax_err < 1e-12 else f"{ax_err:.1e}"
            new_read = VGroup(
                Text(f"N = {N}", font_size=26, color=RED),
                Text(f"axis: largest error {ax_txt}", font_size=18, color=ORANGE),
                Text(f"P: E = {EP:.4f}  (continuous ring {_E_P_REF:.4f})", font_size=18, color=YELLOW),
                Text(f"P: relative error {rel:.1e}", font_size=18, color=YELLOW),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-4.2, -2.95, 0])
            pt = Dot(er_q(i, max(np.log10(rel), -12)), radius=0.07, color=YELLOW)
            if i == 0:
                self.play(FadeIn(new_charges), FadeIn(new_axis), FadeIn(new_read), FadeIn(pt))
            else:
                self.play(FadeOut(charges), FadeIn(new_charges), FadeOut(axis_dots), FadeIn(new_axis),
                          FadeOut(readout), FadeIn(new_read), FadeIn(pt), run_time=1.0)
            charges, axis_dots, readout = new_charges, new_axis, new_read
            err_pts.add(pt)
            self.wait(2.2)
        caption = Text("On the axis every charge adds the same z-component, so any N ≥ 3 is exact.\n"
                       "Near the ring P sees single charges; the error falls like (R/r)^N = (2/3)^N.",
                       font_size=22, line_spacing=0.8).to_edge(UP, buff=0.2)
        self.play(FadeOut(title), FadeIn(caption))
        self.wait(3.0)
