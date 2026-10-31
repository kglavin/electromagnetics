"""Module 4 animations.

HelmholtzFieldLines  two coaxial loops of radius 1 separated by 1, carrying the same current.
                     Field lines are integrated from B = Biot-Savart sum before the scene starts
                     and drawn as curves; a dot runs along each one with a fading trail while
                     the camera rotates. Lines through the centre leave the box and return
                     outside it, lines near the wires close around them.

ChargedParticleMotion  primer 00a, sections 3.3 to 3.5: circles of opposite sense, a helix, the E x B drift
                       of a proton and an electron from rest, and a loop turning into alignment with B.
AmpereLoop             primer 00b, section 3.2: the circulation around deforming loops stays mu0 I, falls to
                       zero when the loop slides off the wire, and counts only enclosed wires.
BoundCurrents          primer 00c, section 3.2: aligned atomic loops cancel on shared edges and leave the rim
                       current K_b = M x n; a non-uniform M leaves J_b = curl M inside.

Render with:
  uv run manim -qh --media_dir media/m4_magnetostatics animations/m4_magnetostatics/scenes.py HelmholtzFieldLines
  uv run manim -qh --media_dir media/m4_magnetostatics animations/m4_magnetostatics/scenes.py ChargedParticleMotion
  uv run manim -qh --media_dir media/m4_magnetostatics animations/m4_magnetostatics/scenes.py AmpereLoop
  uv run manim -qh --media_dir media/m4_magnetostatics animations/m4_magnetostatics/scenes.py BoundCurrents
"""

import numpy as np
from manim import (
    BLUE_B,
    DEGREES,
    DOWN,
    GREY_B,
    ORANGE,
    RED,
    UL,
    WHITE,
    YELLOW,
    Arrow3D,
    Dot3D,
    Text,
    ThreeDScene,
    TracedPath,
    ValueTracker,
    VGroup,
    VMobject,
    linear,
)

from emlib import biot_savart as bs

R = 1.0  # loop radius; the Helmholtz separation equals R
I = 1.0
MID, DL = bs.helmholtz(R, 100)


def B_at(p):
    return bs.wire_B(MID, DL, I, p)


def trace(p0, direction, ds=0.02, n_max=800, half=3.0):
    """Follow dr/ds = +-B/|B| from p0 with RK4. Stop at the box edge or when the line closes."""

    def f(p):
        b = B_at(p)
        return direction * b / np.linalg.norm(b)

    pts = [np.asarray(p0, dtype=float)]
    p = pts[0]
    for k in range(n_max):
        k1 = f(p)
        k2 = f(p + 0.5 * ds * k1)
        k3 = f(p + 0.5 * ds * k2)
        k4 = f(p + ds * k3)
        p = p + ds / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if np.abs(p).max() > half:
            break
        pts.append(p)
        if k > 20 and np.linalg.norm(p - pts[0]) < ds:
            pts.append(pts[0].copy())
            return np.array(pts), True
    return np.array(pts), False


def field_line(p0):
    """A full line through p0: the backward trace reversed, then the forward trace."""
    fwd, closed = trace(p0, +1.0)
    if closed:
        return fwd, True
    bwd, _ = trace(p0, -1.0)
    return np.vstack([bwd[::-1], fwd[1:]]), False


def start_points():
    radii = [0.5, 0.9, 1.4]
    pts = []
    for k, r in enumerate(radii):
        for phi in np.deg2rad([0, 90, 180, 270]) + np.deg2rad(15 * k):
            pts.append([r * np.cos(phi), r * np.sin(phi), 0.0])
    return np.array(pts)


class HelmholtzFieldLines(ThreeDScene):
    def construct(self):
        scale = 1.6  # scene units per metre
        lines_data = [field_line(p) for p in start_points()]

        # The two loops with a marker showing the sense of the current (counter-clockwise about +z).
        coils = VGroup()
        for z in (-R / 2, R / 2):
            ring = VMobject(stroke_color=ORANGE, stroke_width=4)
            t = np.linspace(0, 2 * np.pi, 120)
            ring.set_points_smoothly(scale * np.stack([R * np.cos(t), R * np.sin(t), np.full_like(t, z)], axis=-1))
            marker = Arrow3D(scale * np.array([R, -0.12, z]), scale * np.array([R, 0.22, z]), color=RED,
                             thickness=0.02, height=0.2, base_radius=0.07, resolution=6)
            coils.add(ring, marker)

        lines = VGroup()
        dots = VGroup()
        trails = VGroup()
        for pts, closed in lines_data:
            line = VMobject(stroke_color=BLUE_B if closed else GREY_B, stroke_width=2.0 if closed else 1.5,
                            stroke_opacity=0.9 if closed else 0.5)
            line.set_points_smoothly(scale * pts)
            lines.add(line)
            dot = Dot3D(line.point_from_proportion(0.0), radius=0.05, color=YELLOW)
            dots.add(dot)
            trails.add(TracedPath(dot.get_center, dissipating_time=1.2, stroke_color=YELLOW, stroke_width=3,
                                  stroke_opacity=[0.0, 1.0]))

        t = ValueTracker(0.0)
        for dot, line in zip(dots, lines):
            dot.add_updater(lambda m, line=line: m.move_to(line.point_from_proportion(t.get_value() % 1.0)))

        title = Text("Helmholtz pair: particles following B", font_size=30).to_corner(UL)
        note = Text("blue: closed lines around the wires   grey: lines that leave the box", font_size=22).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(title, note)

        self.set_camera_orientation(phi=68 * DEGREES, theta=-40 * DEGREES, zoom=0.85)
        self.add(coils, lines, trails, dots)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.play(t.animate.set_value(1.0), run_time=12, rate_func=linear)
        self.wait(0.5)


# ---------------------------------------------------------------------------------------------
# Primer scenes (2D). All numbers shown on screen are computed below with NumPy and emlib.
# ---------------------------------------------------------------------------------------------

from manim import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    GREY_C,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Scene,
    Transform,
    Write,
    always_redraw,
    rate_functions,
)

from emlib import integrators as it  # noqa: E402

ORIGIN3 = np.zeros(3)


def _p(x, y):
    return np.array([x, y, 0.0])


def _out_of_page(x, y, r=0.09, colour=GREY_B):
    """The symbol for a vector pointing out of the page: a circle with a dot."""
    return VGroup(Circle(radius=r, color=colour, stroke_width=1.5).move_to(_p(x, y)), Dot(_p(x, y), radius=0.025, color=colour))


def _into_page(x, y, r=0.09, colour=GREY_B):
    """The symbol for a vector pointing into the page: a circle with a cross."""
    d = r * 0.6
    return VGroup(
        Circle(radius=r, color=colour, stroke_width=1.5).move_to(_p(x, y)),
        Line(_p(x - d, y - d), _p(x + d, y + d), color=colour, stroke_width=1.5),
        Line(_p(x - d, y + d), _p(x + d, y - d), color=colour, stroke_width=1.5),
    )


def _b_grid(xs, ys):
    return VGroup(*[_out_of_page(x, y) for x in xs for y in ys])


def _fit(mob, width=13.2):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def _title(text):
    return _fit(Text(text, font_size=30)).to_edge(UP, buff=0.3)


def _caption(text, font_size=22):
    return _fit(Text(text, font_size=font_size)).to_edge(DOWN, buff=0.3)


class ChargedParticleMotion(Scene):
    """Primer M4.0a sections 3.3 and 3.4: circles, helices, the E x B drift, and the torque on a loop.

    Scaled units: |q| = 1, B = 1, proton mass 1, so the cyclotron frequency is 1 and the Larmor
    radius equals the speed. Trajectories come from emlib.integrators.boris.
    """

    def construct(self):
        self.act_circles()
        self.act_helix()
        self.act_drift()
        self.act_torque()

    # (a) uniform B out of the page -------------------------------------------------------
    def act_circles(self):
        title = _title("Uniform B out of the page: F = q v × B turns the velocity, never the speed")
        grid = _b_grid(np.arange(-6.5, 6.6, 1.0), np.arange(-2.0, 2.1, 1.0))
        b_label = Text("B out of the page", font_size=20, color=GREY_B).to_corner(UL, buff=0.25).shift(DOWN * 0.55)
        n_turn, turns = 400, 1.5
        dt = 2 * np.pi / n_turn
        n = int(turns * n_turn)
        scale = 1.5
        runs = []
        for q, colour, centre, sign_text in ((1.0, RED, _p(-3.4, -0.4), "+"), (-1.0, BLUE, _p(3.4, -0.4), "−")):
            x, v = it.boris([0, 0, 0], [1.0, 0, 0], q, 1.0, [0, 0, 0], [0, 0, 1.0], dt, n)
            # start at the top of the circle for q > 0 (centre below), at the bottom for q < 0
            start = centre + _p(0, scale) if q > 0 else centre - _p(0, scale)
            pts = start + scale * x
            F = q * np.cross(v, [0, 0, 1.0])
            runs.append((q, colour, pts, v, F, sign_text))
        tracker = ValueTracker(0)

        def idx():
            return min(int(tracker.get_value()), n)

        movers = VGroup()
        for q, colour, pts, v, F, sign_text in runs:
            path = always_redraw(lambda pts=pts, colour=colour: VMobject(stroke_color=colour, stroke_width=3).set_points_as_corners(pts[: idx() + 1]) if idx() > 0 else VMobject())
            dot = always_redraw(lambda pts=pts, colour=colour: Dot(pts[idx()], radius=0.11, color=colour))
            sign = always_redraw(lambda pts=pts, s=sign_text: Text(s, font_size=22, color=WHITE).move_to(pts[idx()]))
            v_arrow = always_redraw(lambda pts=pts, v=v: Arrow(pts[idx()], pts[idx()] + 1.0 * v[idx()], buff=0, color=WHITE, stroke_width=4, max_tip_length_to_length_ratio=0.25))
            f_arrow = always_redraw(lambda pts=pts, F=F: Arrow(pts[idx()], pts[idx()] + 0.8 * F[idx()], buff=0, color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.25))
            movers.add(path, v_arrow, f_arrow, dot, sign)
        legend = VGroup(
            Text("white: velocity v", font_size=20, color=WHITE),
            Text("green: force q v × B", font_size=20, color=GREEN),
        ).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.15)
        speeds = [np.linalg.norm(r[3], axis=1) for r in runs]
        readout = always_redraw(
            lambda: _fit(Text(
                f"|v| proton {speeds[0][idx()]:.6f}     |v| electron {speeds[1][idx()]:.6f}     (started at 1)",
                font_size=22,
            )).to_edge(DOWN, buff=0.85)
        )
        caption = _caption("positive charge: clockwise    negative charge: counter-clockwise    r = m v / |q| B,  ω = |q| B / m")
        self.play(FadeIn(grid), FadeIn(title), FadeIn(b_label))
        self.play(FadeIn(legend), FadeIn(caption))
        self.add(movers, readout)
        self.play(tracker.animate.set_value(n), run_time=9, rate_func=linear)
        self.wait(1)
        self.play(*[FadeOut(m) for m in (grid, title, b_label, legend, caption, movers, readout)])
        self.remove(movers, readout)

    # (b) helix: velocity component along B ---------------------------------------------------
    def act_helix(self):
        title = _title("A velocity component along B is untouched: the path is a helix")
        n_turn, turns = 300, 3
        dt = 2 * np.pi / n_turn
        n = n_turn * turns
        v_par = 0.25
        # B along +x (to the right on screen); v_perp starts out of the page, so the circle is in the y-z plane.
        x, v = it.boris([0, 0, 0], [v_par, 0, 1.0], 1.0, 1.0, [0, 0, 0], [1.0, 0, 0], dt, n)
        pitch_measured = x[n_turn, 0] - x[0, 0]
        pitch_formula = v_par * 2 * np.pi
        s = 1.15
        side_origin = _p(-5.6, -0.3)
        yc, zc = x[:n_turn, 1].mean(), x[:n_turn, 2].mean()  # guiding centre of the circle
        side = side_origin + s * np.stack([x[:, 0], x[:, 1] - yc, 0 * x[:, 0]], axis=-1)
        end_origin = _p(4.6, -0.3)
        end = end_origin + 1.1 * np.stack([x[:, 1] - yc, x[:, 2] - zc, 0 * x[:, 0]], axis=-1)
        b_arrows = VGroup(*[Arrow(_p(-6.2, y), _p(1.6, y), buff=0, color=GREY_B, stroke_width=2, max_tip_length_to_length_ratio=0.02) for y in (-2.3, 1.7)])
        side_label = Text("side view, B →", font_size=22, color=GREY_B).move_to(_p(-2.3, 2.3))
        end_label = Text("view along B (B out of the page)", font_size=22, color=GREY_B).move_to(_p(4.6, 1.7))
        end_dot = _out_of_page(4.6, -0.3, r=0.12)
        divider = Line(_p(2.3, 2.1), _p(2.3, -2.6), color=GREY_C, stroke_width=1)
        tracker = ValueTracker(0)

        def idx():
            return min(int(tracker.get_value()), n)

        side_path = always_redraw(lambda: VMobject(stroke_color=RED, stroke_width=3).set_points_as_corners(side[: idx() + 1]) if idx() > 0 else VMobject())
        end_path = always_redraw(lambda: VMobject(stroke_color=RED, stroke_width=3).set_points_as_corners(end[: idx() + 1]) if idx() > 0 else VMobject())
        side_dot = always_redraw(lambda: Dot(side[idx()], radius=0.1, color=RED))
        end_dot_p = always_redraw(lambda: Dot(end[idx()], radius=0.1, color=RED))
        readout = _fit(Text(
            f"pitch per turn: measured {pitch_measured:.4f},  v∥ T = 2π v∥/ω = {pitch_formula:.4f}   (v∥ = {v_par}, v⊥ = 1, units with ω = 1)",
            font_size=20,
        )).to_edge(DOWN, buff=0.85)
        caption = _caption("v∥ feels no force and stays constant; v⊥ circles exactly as before")
        self.play(FadeIn(title), FadeIn(b_arrows), FadeIn(side_label), FadeIn(end_label), FadeIn(end_dot), Create(divider))
        self.add(side_path, end_path, side_dot, end_dot_p)
        self.play(tracker.animate.set_value(n), run_time=8, rate_func=linear)
        self.play(FadeIn(readout), FadeIn(caption))
        self.wait(1.5)
        everything = (title, b_arrows, side_label, end_label, end_dot, divider, side_path, end_path, side_dot, end_dot_p, readout, caption)
        self.play(*[FadeOut(m) for m in everything])
        self.remove(*everything)

    # (c) crossed E and B: the drift ----------------------------------------------------------------
    def act_drift(self):
        title = _title("Crossed fields: E along +x, B out of the page. Both charges drift along E × B")
        E0, B0 = 0.5, 1.0
        m_p, m_e = 1.0, 0.3  # electron mass scaled up so its small cycloid is visible
        t_end = 4 * np.pi  # two proton periods
        n = 1600
        dt = t_end / n
        xp, _ = it.boris([0, 0, 0], [0, 0, 0], 1.0, m_p, [E0, 0, 0], [0, 0, B0], dt, n)
        xe, _ = it.boris([0, 0, 0], [0, 0, 0], -1.0, m_e, [E0, 0, 0], [0, 0, B0], dt, n)
        # measured drift: displacement over a whole number of each particle's own periods
        Tp, Te = 2 * np.pi * m_p / B0, 2 * np.pi * m_e / B0
        kp = int(round(2 * Tp / dt))
        ke = int(round(int(t_end / Te) * Te / dt))
        drift_p = (xp[kp] - xp[0]) / (kp * dt)
        drift_e = (xe[ke] - xe[0]) / (ke * dt)
        s = 0.55
        p_pts = _p(-1.6, 2.0) + s * xp
        e_pts = _p(1.6, 2.0) + s * xe
        e_arrows = VGroup(*[Arrow(_p(x0, y), _p(x0 + 0.8, y), buff=0, color=YELLOW, stroke_width=3, max_tip_length_to_length_ratio=0.3) for x0 in (-6.4, -5.0, 4.4, 5.8) for y in (1.6, 0.4, -0.8, -2.0)])
        grid = _b_grid(np.arange(-3.5, 3.6, 1.0), np.arange(-2.5, 1.6, 1.0))
        labels = VGroup(
            Text("E", font_size=24, color=YELLOW).move_to(_p(-5.3, 2.35)),
            Text("E", font_size=24, color=YELLOW).move_to(_p(5.1, 2.35)),
            Text("proton", font_size=22, color=RED).move_to(_p(-1.6, 2.55)),
            Text("electron (mass scaled up)", font_size=22, color=BLUE).move_to(_p(1.6, 2.55)),
        )
        tracker = ValueTracker(0)

        def idx():
            return min(int(tracker.get_value()), n)

        movers = VGroup(
            always_redraw(lambda: VMobject(stroke_color=RED, stroke_width=3).set_points_as_corners(p_pts[: idx() + 1]) if idx() > 0 else VMobject()),
            always_redraw(lambda: VMobject(stroke_color=BLUE, stroke_width=3).set_points_as_corners(e_pts[: idx() + 1]) if idx() > 0 else VMobject()),
            always_redraw(lambda: Dot(p_pts[idx()], radius=0.1, color=RED)),
            always_redraw(lambda: Dot(e_pts[idx()], radius=0.08, color=BLUE)),
        )
        drift_arrow = Arrow(_p(0, 1.0), _p(0, -1.2), buff=0, color=WHITE, stroke_width=5)
        drift_label = Text("E × B", font_size=24).next_to(drift_arrow, RIGHT, buff=0.1)
        readout = _fit(Text(
            f"measured drift speed: proton {np.linalg.norm(drift_p):.4f}, electron {np.linalg.norm(drift_e):.4f}    E/B = {E0 / B0:.4f}    (both along −y)",
            font_size=20,
        )).to_edge(DOWN, buff=0.85)
        caption = _caption("start from rest; v_drift = E × B / B² for every charge and mass: the paths are cycloids")
        self.play(FadeIn(title), FadeIn(grid), FadeIn(e_arrows), FadeIn(labels))
        self.add(movers)
        self.play(tracker.animate.set_value(n), run_time=10, rate_func=linear)
        self.play(FadeIn(drift_arrow), FadeIn(drift_label), FadeIn(readout), FadeIn(caption))
        self.wait(2)
        everything = (title, grid, e_arrows, labels, movers, drift_arrow, drift_label, readout, caption)
        self.play(*[FadeOut(m) for m in everything])
        self.remove(*everything)

    # (d) torque on a current loop ------------------------------------------------------------------
    def act_torque(self):
        title = _title("A current loop in B: the torque m × B turns the moment towards B")
        # damped rigid rotation: J theta'' = -m B sin(theta) - c theta'
        theta, omega, dt = np.radians(110.0), 0.0, 0.01
        thetas = [theta]
        for _ in range(1200):
            acc = -np.sin(theta) - 0.6 * omega
            omega += acc * dt
            theta += omega * dt
            thetas.append(theta)
        thetas = np.array(thetas)
        energies = -np.cos(thetas)  # U / (m B)
        b_arrows = VGroup(*[Arrow(_p(-6, y), _p(-0.4, y), buff=0, color=GREY_B, stroke_width=2, max_tip_length_to_length_ratio=0.03) for y in (-2.0, -1.0, 0.0, 1.0, 2.0)])
        b_label = Text("B", font_size=26, color=GREY_B).move_to(_p(-0.1, 2.0))
        centre = _p(-3.2, 0.0)
        tracker = ValueTracker(0)

        def idx():
            return min(int(tracker.get_value()), len(thetas) - 1)

        def loop_group():
            th = thetas[idx()]
            m_dir = np.array([np.cos(th), np.sin(th), 0.0])
            edge = np.array([-np.sin(th), np.cos(th), 0.0])  # the loop seen edge-on lies along this line
            a, b = centre + 1.2 * edge, centre - 1.2 * edge
            return VGroup(
                Line(a, b, color=ORANGE, stroke_width=8),
                _out_of_page(*a[:2], r=0.13, colour=ORANGE),
                _into_page(*b[:2], r=0.13, colour=ORANGE),
                Arrow(centre, centre + 1.6 * m_dir, buff=0, color=RED, stroke_width=6),
                Text("m", font_size=26, color=RED).move_to(centre + 1.95 * m_dir),
            )

        loop = always_redraw(loop_group)
        readout = always_redraw(
            lambda: VGroup(
                Text(f"angle between m and B: {np.degrees(thetas[idx()]):6.1f}°", font_size=24),
                Text(f"U = −m·B = {energies[idx()]:+.3f} m B", font_size=24),
                Text(f"|τ| = |m × B| = {abs(np.sin(thetas[idx()])):.3f} m B", font_size=24),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(_p(3.6, 0.3))
        )
        caption = _caption("loop seen edge-on: current out of the page at the top, into it at the bottom (damped for clarity)")
        self.play(FadeIn(title), FadeIn(b_arrows), FadeIn(b_label), FadeIn(caption))
        self.add(loop, readout)
        self.play(tracker.animate.set_value(len(thetas) - 1), run_time=7, rate_func=linear)
        self.wait(1.5)


class AmpereLoop(Scene):
    """Primer M4.0b section 3.2: the circulation of B around a closed loop counts the enclosed current.

    Units: B = mu0 I / (2 pi r) phi_hat with mu0 I = 1, so the circulation is in units of mu0 I.
    Every number is a midpoint sum of B . dl around the polyline that is drawn.
    """

    N = 720
    SCALE = 1.0
    CENTRE = _p(-0.5, 0.25)

    @staticmethod
    def field(points, wires):
        """B at (N, 2) points from line currents [(x, y, I)], units mu0 = 1."""
        B = np.zeros_like(points)
        for wx, wy, I in wires:
            d = points - np.array([wx, wy])
            r2 = np.sum(d**2, axis=1)
            B += I / (2 * np.pi * r2)[:, None] * np.stack([-d[:, 1], d[:, 0]], axis=1)
        return B

    def running_circulation(self, loop_xy, wires):
        """Cumulative midpoint sum of B . dl around a closed polyline (N, 2). Returns (N + 1,) array."""
        closed = np.vstack([loop_xy, loop_xy[:1]])
        dl = np.diff(closed, axis=0)
        mid = 0.5 * (closed[1:] + closed[:-1])
        return np.concatenate([[0.0], np.cumsum(np.sum(self.field(mid, wires) * dl, axis=1))])

    def shape(self, name, offset=(0.0, 0.0)):
        """Loop shapes in polar form r(theta) about the wire, sampled at the same N angles."""
        th = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        if name == "circle":
            r = 1.5 + 0 * th
        elif name == "wobbly":
            r = 1.55 + 0.45 * np.sin(5 * th) + 0.15 * np.cos(3 * th)
        elif name == "squashed":
            a, b = 2.4, 0.8
            r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
        elif name == "square":
            r = 1.5 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1) + np.array(offset)

    def to_screen(self, xy):
        return self.CENTRE + self.SCALE * np.column_stack([xy, np.zeros(len(xy))])

    def construct(self):
        wire = [(0.0, 0.0, 1.0)]
        title = _title("Ampere's law: ∮ B·dl = μ0 I_enc for every loop around the wire")
        field_lines = VGroup()
        for k_r, r in enumerate((0.6, 1.1, 1.9, 2.6)):
            c = Circle(radius=r, color=YELLOW, stroke_width=1.5, stroke_opacity=0.45).move_to(self.CENTRE)
            field_lines.add(c)
            for ang in np.radians(20 + 37 * k_r + np.array([0, 120, 240])):
                pos = self.CENTRE + _p(r * np.cos(ang), r * np.sin(ang))
                tang = _p(-np.sin(ang), np.cos(ang))
                field_lines.add(Arrow(pos - 0.12 * tang, pos + 0.12 * tang, buff=0, color=YELLOW, stroke_width=2, max_tip_length_to_length_ratio=0.9).set_opacity(0.6))
        wire_sym = _out_of_page(*self.CENTRE[:2], r=0.16, colour=ORANGE)
        wire_label = Text("I out of the page", font_size=20, color=ORANGE).move_to(_p(-5.0, 2.6))
        field_lines.add(Text("B (yellow) circles the wire", font_size=20, color=YELLOW).move_to(_p(-5.0, 2.2)))
        self.play(FadeIn(title), FadeIn(field_lines), FadeIn(wire_sym), FadeIn(wire_label))

        names = ["circle", "wobbly", "squashed", "square"]
        shapes = {nm: self.shape(nm) for nm in names}
        circ = {nm: self.running_circulation(shapes[nm], wire) for nm in names}

        loop_tracker = ValueTracker(0.0)  # morph parameter between consecutive shapes
        run_tracker = ValueTracker(0.0)  # position of the dot around the loop, 0..N
        state = {"from": "circle", "to": "circle", "wires": wire, "offset": np.zeros(2)}

        def current_xy():
            s = loop_tracker.get_value()
            return (1 - s) * shapes[state["from"]] + s * shapes[state["to"]] + state["offset"]

        def current_run():
            return self.running_circulation(current_xy(), state["wires"])

        loop_mob = always_redraw(lambda: VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(self.to_screen(np.vstack([current_xy(), current_xy()[:1]]))))

        def dot_mob():
            k = int(run_tracker.get_value()) % self.N
            return Dot(self.to_screen(current_xy())[k], radius=0.09, color=GREEN)

        dot = always_redraw(dot_mob)

        def readout_mob():
            k = int(run_tracker.get_value())
            run = current_run()
            k = min(k, self.N)
            return Text(f"running ∮ B·dl  =  {run[k]:+.3f}  μ0 I", font_size=28).to_edge(DOWN, buff=0.85)

        readout = always_redraw(readout_mob)
        caption = _caption("the dot walks counter-clockwise; the sum is the midpoint rule over the drawn loop")
        self.play(Create(loop_mob), FadeIn(caption))
        self.add(dot, readout)

        def walk(run_time=3.0):
            run_tracker.set_value(0)
            self.play(run_tracker.animate.set_value(self.N), run_time=run_time, rate_func=linear)
            self.wait(0.6)

        walk()
        for nxt in names[1:]:
            state["from"], state["to"] = state["to"], nxt
            loop_tracker.set_value(0.0)
            run_tracker.set_value(self.N)  # while morphing, show the full circulation of the closed loop
            self.play(loop_tracker.animate.set_value(1.0), run_time=1.5)
            state["from"] = nxt
            walk(2.5)

        # slide the loop off the wire: the readout is hidden while the edge crosses the wire
        self.remove(readout)
        run_tracker.set_value(0)
        slide = ValueTracker(0.0)
        base_offset = np.zeros(2)

        def set_offset(m):
            state["offset"] = base_offset + np.array([slide.get_value(), 0.0])

        loop_mob.add_updater(set_offset)
        self.play(slide.animate.set_value(3.6), run_time=2.5)
        loop_mob.remove_updater(set_offset)
        state["offset"] = np.array([3.6, 0.0])
        self.add(readout)
        new_caption = _caption("the wire is outside: what the field adds on the near side it takes back on the far side")
        self.play(Transform(caption, new_caption))
        walk(3.0)
        self.wait(0.5)

        # worked example 4: three wires, unit circle around the origin (positions scaled by 2 on screen)
        self.play(*[FadeOut(m) for m in (loop_mob, dot, readout, field_lines, wire_sym, wire_label, caption)])
        self.remove(loop_mob, dot, readout)
        wires3 = [(0.3, 0.0, 3.0), (-0.3, 0.2, -1.0), (1.5, 0.0, 2.0)]
        k = 2.0  # screen units per metre
        th = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        loop3 = np.stack([np.cos(th), np.sin(th)], axis=1)
        run3 = self.running_circulation(loop3, wires3)
        centre3 = _p(-1.2, 0.2)

        def scr(xy):
            return centre3 + k * np.column_stack([xy, np.zeros(len(xy))])

        title3 = _title("Several wires: only the enclosed current counts")
        syms = VGroup()
        for wx, wy, I in wires3:
            pos = centre3 + k * _p(wx, wy)
            sym = _out_of_page(*pos[:2], r=0.16, colour=ORANGE) if I > 0 else _into_page(*pos[:2], r=0.16, colour=ORANGE)
            lab = Text(f"{I:+.0f} A", font_size=22, color=ORANGE).next_to(sym, UP, buff=0.12)
            syms.add(sym, lab)
        loop3_mob = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(scr(np.vstack([loop3, loop3[:1]])))
        run_tracker.set_value(0)
        dot3 = always_redraw(lambda: Dot(scr(loop3)[int(run_tracker.get_value()) % self.N], radius=0.09, color=GREEN))
        readout3 = always_redraw(lambda: Text(f"running ∮ B·dl / μ0  =  {run3[min(int(run_tracker.get_value()), self.N)]:+.3f} A", font_size=28).to_edge(DOWN, buff=0.85))
        caption3 = _caption("enclosed: +3 A and −1 A, net 2 A.  The +2 A wire outside changes B on the loop but not the total")
        self.play(Transform(title, title3), FadeIn(syms), Create(loop3_mob), FadeIn(caption3))
        self.add(dot3, readout3)
        self.play(run_tracker.animate.set_value(self.N), run_time=4, rate_func=linear)
        self.wait(2)


class BoundCurrents(Scene):
    """Primer M4.0c section 3.2: aligned atomic current loops cancel inside and leave a surface current.

    Part 1: uniform M. Loops with random senses align counter-clockwise (M out of the page); the
    currents on every shared edge are equal and opposite and cancel in pairs; the rim current
    K_b = M x n remains. Part 2: M grows from left to right; the shared vertical edges keep a
    net downward current, J_b = curl M.
    """

    COLS, ROWS = 6, 4
    A = 0.9  # loop size on screen
    GAP = 0.08  # half the space between neighbouring loops

    def loop_centre(self, i, j, origin):
        return origin + _p((i - (self.COLS - 1) / 2) * self.A, (j - (self.ROWS - 1) / 2) * self.A)

    def edge_arrows(self, centre, sense, colour=WHITE, width=3.0, opacity=1.0):
        """Four arrows around a square loop. sense = +1 counter-clockwise, -1 clockwise.

        Returned in the order bottom, right, top, left (for the ccw sense the bottom runs +x).
        """
        h = self.A / 2 - self.GAP
        corners = [centre + _p(-h, -h), centre + _p(h, -h), centre + _p(h, h), centre + _p(-h, h)]
        arrows = []
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            if sense < 0:
                a, b = b, a
            arrows.append(Arrow(a, b, buff=0.04, color=colour, stroke_width=width, max_tip_length_to_length_ratio=0.22).set_opacity(opacity))
        return arrows

    def construct(self):
        rng = np.random.default_rng(3)
        origin = _p(-0.2, 0.15)
        title = _title("Magnetization as atomic current loops")
        senses = rng.choice([-1, 1], size=(self.COLS, self.ROWS))
        loops = {}
        group = VGroup()
        for i in range(self.COLS):
            for j in range(self.ROWS):
                arrows = self.edge_arrows(self.loop_centre(i, j, origin), senses[i, j])
                loops[(i, j)] = VGroup(*arrows)
                group.add(loops[(i, j)])
        body = Rectangle(width=self.COLS * self.A + 0.1, height=self.ROWS * self.A + 0.1, color=GREY, stroke_width=2).move_to(origin)
        caption = _caption("random senses: the moments cancel and M = 0")
        self.play(FadeIn(title), Create(body), FadeIn(group), FadeIn(caption))
        self.wait(1)

        # align: every clockwise loop flips to counter-clockwise
        flips = []
        for (i, j), g in loops.items():
            if senses[i, j] < 0:
                target = VGroup(*self.edge_arrows(self.loop_centre(i, j, origin), +1))
                flips.append(Transform(g, target))
        m_symbols = VGroup(*[_out_of_page(*self.loop_centre(i, j, origin)[:2], r=0.1, colour=RED) for i in range(self.COLS) for j in range(self.ROWS)])
        new_caption = _caption("an applied field aligns them: every loop counter-clockwise, M out of the page (red)")
        self.play(*flips, Transform(caption, new_caption), run_time=2)
        self.play(FadeIn(m_symbols))
        self.wait(1)

        # pairs of shared edges cancel. Edge order per ccw loop: 0 bottom (+x), 1 right (+y), 2 top (-x), 3 left (-y)
        internal = []
        for i in range(self.COLS):
            for j in range(self.ROWS):
                if i + 1 < self.COLS:
                    internal.append((loops[(i, j)][1], loops[(i + 1, j)][3]))
                if j + 1 < self.ROWS:
                    internal.append((loops[(i, j)][2], loops[(i, j + 1)][0]))
        cancel_caption = _caption("each shared edge carries two equal currents in opposite directions: they cancel")
        self.play(Transform(caption, cancel_caption))
        highlight = [m.animate.set_color(GREY_C) for pair in internal for m in pair]
        self.play(*highlight, run_time=1.5)
        self.play(*[FadeOut(m) for pair in internal for m in pair], run_time=2)

        # what is left: the rim. Replace it by one continuous circulating current.
        rim_w, rim_h = self.COLS * self.A - 2 * self.GAP, self.ROWS * self.A - 2 * self.GAP
        rim = Rectangle(width=rim_w, height=rim_h, color=RED, stroke_width=5).move_to(origin)
        mids = [origin + _p(0, -rim_h / 2), origin + _p(rim_w / 2, 0), origin + _p(0, rim_h / 2), origin + _p(-rim_w / 2, 0)]
        dirs = [_p(1, 0), _p(0, 1), _p(-1, 0), _p(0, -1)]
        rim_arrows = VGroup(*[Arrow(c - 0.35 * d, c + 0.35 * d, buff=0, color=RED, stroke_width=6, max_tip_length_to_length_ratio=0.4) for c, d in zip(mids, dirs)])
        kb_label = Text("K_b = M × n", font_size=26, color=RED).next_to(rim, RIGHT, buff=0.3)
        rest = VGroup(*[a for g in loops.values() for a in g if a not in {m for pair in internal for m in pair}])
        final_caption = _caption("only the rim current survives: a uniformly magnetized bar is a solenoid with n I = M")
        self.play(Transform(rest, VGroup(rim, rim_arrows)), FadeIn(kb_label), Transform(caption, final_caption), run_time=2)
        self.wait(2)
        everything = (title, body, rest, m_symbols, kb_label, caption)
        self.play(*[FadeOut(m) for m in everything])
        self.remove(*everything)

        # Part 2: M_z grows from left to right, strength i + 1 in column i.
        title2 = _title("Non-uniform M: the cancellation is incomplete")
        strengths = np.arange(1, self.COLS + 1)
        loops2 = {}
        group2 = VGroup()
        for i in range(self.COLS):
            w = 1.2 + 0.9 * i
            for j in range(self.ROWS):
                g = VGroup(*self.edge_arrows(self.loop_centre(i, j, origin), +1, width=w))
                loops2[(i, j)] = g
                group2.add(g)
        numbers = VGroup(*[Text(str(strengths[i]), font_size=20, color=RED).move_to(self.loop_centre(i, j, origin)) for i in range(self.COLS) for j in range(self.ROWS)])
        body2 = Rectangle(width=self.COLS * self.A + 0.1, height=self.ROWS * self.A + 0.1, color=GREY, stroke_width=2).move_to(origin)
        cap2 = _caption("loop current proportional to M_z, which grows with x (numbers: M_z in units of M0)")
        self.play(FadeIn(title2), Create(body2), FadeIn(group2), FadeIn(numbers), FadeIn(cap2))
        self.wait(1)
        # horizontal shared edges: same column, same strength, cancel. Vertical shared edges: right edge of
        # column i (+y, strength i+1) and left edge of column i+1 (-y, strength i+2): net 1 along -y.
        horiz = [(loops2[(i, j)][2], loops2[(i, j + 1)][0]) for i in range(self.COLS) for j in range(self.ROWS - 1)]
        vert = [(loops2[(i, j)][1], loops2[(i + 1, j)][3]) for i in range(self.COLS - 1) for j in range(self.ROWS)]
        self.play(Transform(cap2, _caption("horizontal shared edges: equal and opposite, they cancel")))
        self.play(*[FadeOut(m) for pair in horiz for m in pair], run_time=1.5)
        net_arrows = VGroup()
        for i in range(self.COLS - 1):
            for j in range(self.ROWS):
                x = origin[0] + (i + 0.5 - (self.COLS - 1) / 2) * self.A
                y = origin[1] + (j - (self.ROWS - 1) / 2) * self.A
                net_arrows.add(Arrow(_p(x, y + 0.3), _p(x, y - 0.3), buff=0, color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.35))
        net_net = int(strengths[1] - strengths[0])
        self.play(
            Transform(cap2, _caption(f"vertical shared edges: {strengths[1]} down minus {strengths[0]} up leaves a net {net_net} flowing down (−y)")),
            *[Transform(VGroup(*pair), net_arrows[k]) for k, pair in enumerate(vert)],
            run_time=2,
        )
        jb_label = Text("green: J_b = ∇ × M = −(∂M_z/∂x) ŷ", font_size=26, color=GREEN).next_to(body2, DOWN, buff=0.35)
        self.play(FadeIn(jb_label), Transform(cap2, _caption("a net volume current remains inside, and it closes through the rim currents")))
        self.wait(2.5)
