"""Module 5 animations: current, circuits and disordered media.

DriftCurrent         primer 00a: electrons jitter with fast thermal motion; a field adds a slow drift against E,
                     and a cross-section counts the net charge that passes. Speeds for copper are computed here.
CapacitorCharging    primer 00b: an RC loop; the switch closes, the current starts at V0/R and decays while the
                     capacitor charges as 1 - exp(-t/RC); markers at t = RC and 5 RC.
PhasorRLC            primer 00c and notebook 03: phasors of a series RLC drawn head to tail and rotating, their
                     projections traced as sines; then a frequency sweep through resonance.
PercolationBackbone  primer 00d and notebook 04: a 20 by 20 bond lattice between two electrodes; p rises from
                     0.3 to 0.7, a spanning cluster forms and the current flows on its backbone.

Render with:
  uv run manim -qh --media_dir media/m5_circuits animations/m5_circuits/scenes.py DriftCurrent
  uv run manim -qh --media_dir media/m5_circuits animations/m5_circuits/scenes.py CapacitorCharging
  uv run manim -qh --media_dir media/m5_circuits animations/m5_circuits/scenes.py PhasorRLC
  uv run manim -qh --media_dir media/m5_circuits animations/m5_circuits/scenes.py PercolationBackbone
"""

import numpy as np
from manim import (
    BLUE,
    BLUE_B,
    DOWN,
    GREEN,
    GREY,
    GREY_B,
    GREY_D,
    LEFT,
    ORANGE,
    PURPLE_B,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Scene,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    interpolate_color,
    linear,
)

from emlib import circuits as ci
from emlib import percolation as pc


def _p(x, y):
    return np.array([x, y, 0.0])


def _fit(mob, width=13.2):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def _title(text):
    return _fit(Text(text, font_size=30)).to_edge(UP, buff=0.3)


def _caption(text, font_size=20):
    return _fit(Text(text, font_size=font_size)).to_edge(DOWN, buff=0.3)


def _plain_axes(x_range, y_range, x_length, y_length):
    return Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        axis_config={"include_numbers": False, "include_tip": False, "stroke_width": 2, "color": GREY_B},
    )


# ---------------------------------------------------------------------------------------------
# DriftCurrent


class DriftCurrent(Scene):
    """Primer M5.0a: thermal motion carries no current, a small drift does.

    Copper numbers: n = 8.5e28 /m^3, sigma = 5.96e7 S/m, T = 300 K, J = 1 A/mm^2.
    Drift speed v_d = J / (n e); classical thermal speed sqrt(3 k T / m); Drude collision time
    tau = m sigma / (n e^2); field E = J / sigma. On screen the drift is exaggerated by many orders
    of magnitude so it can be seen at all.
    """

    E_CHARGE, M_E, K_B = 1.602e-19, 9.109e-31, 1.381e-23
    N_CU, SIGMA_CU, T = 8.5e28, 5.96e7, 300.0
    J = 1e6  # A/m^2 = 1 A/mm^2

    def physics(self):
        v_d = self.J / (self.N_CU * self.E_CHARGE)
        v_th = np.sqrt(3 * self.K_B * self.T / self.M_E)
        tau = self.M_E * self.SIGMA_CU / (self.N_CU * self.E_CHARGE**2)
        E = self.J / self.SIGMA_CU
        return v_d, v_th, tau, E

    def simulate(self, n=60, t_quiet=6.0, t_drift=10.0, dt=1 / 60, seed=4):
        """Random walk in a box with periodic x and reflecting y; drift switched on after t_quiet.

        Returns positions (K, n, 2) in screen units and the cumulative net count of electrons that
        crossed x = 0 moving in -x (which is conventional current along +x).
        """
        rng = np.random.default_rng(seed)
        half_w, half_h = 5.0, 1.15
        speed, t_coll, drift = 2.2, 0.22, 0.7
        K = int((t_quiet + t_drift) / dt) + 1
        pos = np.empty((K, n, 2))
        pos[0] = np.column_stack([rng.uniform(-half_w, half_w, n), rng.uniform(-half_h, half_h, n)])
        ang = rng.uniform(0, 2 * np.pi, n)
        vel = speed * np.column_stack([np.cos(ang), np.sin(ang)])
        count = np.zeros(K)
        for k in range(1, K):
            t = k * dt
            hit = rng.random(n) < dt / t_coll  # collisions randomise the direction
            ang = rng.uniform(0, 2 * np.pi, hit.sum())
            vel[hit] = speed * np.column_stack([np.cos(ang), np.sin(ang)])
            v = vel.copy()
            if t > t_quiet:
                v[:, 0] -= drift  # electrons drift against E (E along +x)
            new = pos[k - 1] + v * dt
            lo, hi = new[:, 1] < -half_h, new[:, 1] > half_h
            new[lo, 1] = -2 * half_h - new[lo, 1]
            new[hi, 1] = 2 * half_h - new[hi, 1]
            vel[lo | hi, 1] *= -1
            old_x = pos[k - 1, :, 0]
            crossed_left = np.sum((old_x > 0) & (new[:, 0] <= 0))
            crossed_right = np.sum((old_x <= 0) & (new[:, 0] > 0))
            count[k] = count[k - 1] + crossed_left - crossed_right
            new[:, 0] = (new[:, 0] + half_w) % (2 * half_w) - half_w
            pos[k] = new
        return pos, count, K, dt, t_quiet

    def construct(self):
        v_d, v_th, tau, E = self.physics()
        pos, count, K, dt, t_quiet = self.simulate()
        centre = _p(0, 0.35)

        title = _title("Current is a slow drift on top of fast random motion")
        wire = Rectangle(width=10.6, height=2.7, color=GREY_B, stroke_width=3).move_to(centre)
        ions = VGroup(*[Dot(centre + _p(x, y), radius=0.05, color=GREY_D) for x in np.arange(-4.8, 4.9, 0.8) for y in (-0.9, -0.3, 0.3, 0.9)])
        plane = DashedLine(centre + _p(0, -1.45), centre + _p(0, 1.45), color=GREEN, stroke_width=3)
        plane_label = Text("cross-section", font_size=20, color=GREEN).next_to(plane, UP, buff=0.08)
        self.play(FadeIn(title), Create(wire), FadeIn(ions), Create(plane), FadeIn(plane_label))

        frame = ValueTracker(0)

        def k_now():
            return min(int(frame.get_value()), K - 1)

        electrons = always_redraw(lambda: VGroup(*[Dot(centre + _p(x, y), radius=0.07, color=BLUE_B) for x, y in pos[k_now()]]))
        readout = always_redraw(
            lambda: Text(f"net electrons through the cross-section, right to left: {int(count[k_now()]):+d}", font_size=22).next_to(wire, DOWN, buff=0.25)
        )
        state = Text("no field: random motion, no net flow", font_size=22, color=GREY_B).move_to(centre + _p(0, -2.15))
        self.add(electrons, readout)
        self.play(FadeIn(state))
        k_switch = int(t_quiet / dt)
        self.play(frame.animate.set_value(k_switch), run_time=t_quiet, rate_func=linear)

        e_arrow = Arrow(centre + _p(-5.2, 2.15), centre + _p(-3.4, 2.15), buff=0, color=YELLOW, stroke_width=5)
        e_label = Text("E", font_size=26, color=YELLOW).next_to(e_arrow, RIGHT, buff=0.12)
        i_label = Text("conventional current I → (electrons drift ←)", font_size=22, color=YELLOW).next_to(e_label, RIGHT, buff=0.4)
        new_state = Text("field on: the same jitter plus a slow drift against E", font_size=22, color=YELLOW).move_to(centre + _p(0, -2.15))
        self.play(FadeOut(state), FadeIn(e_arrow), FadeIn(e_label), FadeIn(i_label), FadeIn(new_state), run_time=0.8)
        self.play(frame.animate.set_value(K - 1), run_time=(K - 1 - k_switch) * dt, rate_func=linear)

        numbers = VGroup(
            Text(f"copper at 1 A/mm²:  drift v_d = J / n e = {v_d:.1e} m/s     thermal √(3kT/m) = {v_th:.1e} m/s", font_size=20),
            Text(f"ratio {v_d / v_th:.0e};  E = J/σ = {E:.3f} V/m;  time between collisions τ = mσ/ne² = {tau:.1e} s.  Drift exaggerated on screen.", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        _fit(numbers).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(new_state), FadeIn(numbers))
        self.wait(3)


# ---------------------------------------------------------------------------------------------
# CapacitorCharging


class CapacitorCharging(Scene):
    """Primer M5.0b: R = 1 kOhm, C = 1 mF (RC = 1 s), V0 = 5 V. Curves from emlib.circuits.rc_charging."""

    R, C, V0 = 1e3, 1e-3, 5.0

    def circuit(self):
        x0, x1, y0, y1 = -6.0, -2.0, -1.6, 1.6
        g = VGroup()
        # battery on the left side, long plate is +
        g.add(Line(_p(x0, y0), _p(x0, -0.25)), Line(_p(x0, 0.25), _p(x0, y1)))
        g.add(Line(_p(x0 - 0.4, 0.25), _p(x0 + 0.4, 0.25), stroke_width=5), Line(_p(x0 - 0.22, -0.25), _p(x0 + 0.22, -0.25), stroke_width=7))
        g.add(Text("+", font_size=24).move_to(_p(x0 - 0.6, 0.45)), Text("V0 = 5 V", font_size=22).move_to(_p(x0 + 1.0, 0.0)))
        # resistor zigzag on the top
        zx = np.linspace(x0 + 1.2, x1 - 1.2, 13)
        zy = y1 + 0.22 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0])
        g.add(Line(_p(x0, y1), _p(zx[0], y1)), VMobject().set_points_as_corners([_p(a, b) for a, b in zip(zx, zy)]), Line(_p(zx[-1], y1), _p(x1, y1)))
        g.add(Text("R = 1 kΩ", font_size=22).move_to(_p(0.5 * (x0 + x1), y1 + 0.6)))
        # capacitor on the right side
        g.add(Line(_p(x1, y1), _p(x1, 0.18)), Line(_p(x1, -0.18), _p(x1, y0)))
        g.add(Line(_p(x1 - 0.55, 0.18), _p(x1 + 0.55, 0.18), stroke_width=6), Line(_p(x1 - 0.55, -0.18), _p(x1 + 0.55, -0.18), stroke_width=6))
        g.add(Text("C = 1 mF", font_size=22).move_to(_p(x1 + 1.4, 0.0)))
        # bottom wire with a switch
        g.add(Line(_p(x0, y0), _p(-4.5, y0)), Line(_p(-3.5, y0), _p(x1, y0)), Dot(_p(-4.5, y0), radius=0.06), Dot(_p(-3.5, y0), radius=0.06))
        return g.set_color(WHITE), (x0, x1, y0, y1)

    def construct(self):
        R, C, V0 = self.R, self.C, self.V0
        tau = R * C
        I0 = V0 / R
        title = _title("Charging a capacitor through a resistor: time constant RC = 1 s")
        circuit, (x0, x1, y0, y1) = self.circuit()
        switch_open = Line(_p(-4.5, y0), _p(-3.6, y0 + 0.6), color=WHITE)
        switch_closed = Line(_p(-4.5, y0), _p(-3.5, y0), color=WHITE)
        switch_label = Text("switch", font_size=20, color=GREY_B).move_to(_p(-4.0, y0 - 0.4))

        axes = _plain_axes([0, 5, 1], [0, 1.05, 0.5], 5.6, 3.6).move_to(_p(3.2, 0.1))
        t_label = Text("t / RC", font_size=20, color=GREY_B).next_to(axes.x_axis, RIGHT, buff=0.15)
        ticks = VGroup(*[Text(str(k), font_size=18, color=GREY_B).next_to(axes.c2p(k, 0), DOWN, buff=0.12) for k in range(6)])
        yt = VGroup(Text("1", font_size=18, color=GREY_B).next_to(axes.c2p(0, 1), LEFT, buff=0.12), Text("0", font_size=18, color=GREY_B).next_to(axes.c2p(0, 0), LEFT, buff=0.12))
        legend = VGroup(Text("V_C / V0", font_size=20, color=YELLOW), Text("I / I0", font_size=20, color=BLUE)).arrange(RIGHT, buff=0.5).next_to(axes, UP, buff=0.15)
        self.play(FadeIn(title), Create(circuit), Create(switch_open), FadeIn(switch_label), Create(axes), FadeIn(t_label), FadeIn(ticks), FadeIn(yt), FadeIn(legend))
        self.play(Transform(switch_open, switch_closed), run_time=0.6)

        t_end = 5.0 * tau
        clock = ValueTracker(0.0)

        def Vc(t):
            return float(ci.rc_charging(t, R, C, V0))

        def I(t):
            return I0 * np.exp(-t / tau)

        v_curve = always_redraw(lambda: axes.plot(lambda s: Vc(s * tau) / V0, x_range=[0, max(clock.get_value() / tau, 1e-3)], color=YELLOW, stroke_width=4))
        i_curve = always_redraw(lambda: axes.plot(lambda s: I(s * tau) / I0, x_range=[0, max(clock.get_value() / tau, 1e-3)], color=BLUE, stroke_width=4))

        def current_arrows():
            frac = I(clock.get_value()) / I0
            g = VGroup()
            for a, b in ((_p(x0 + 0.3, y1 - 0.35), _p(x0 + 1.0, y1 - 0.35)), (_p(x1 - 0.35, y1 - 0.3), _p(x1 - 0.35, 0.7)), (_p(x1 - 0.8, y0 + 0.35), _p(x1 - 1.5, y0 + 0.35)), (_p(x0 + 0.35, -0.8), _p(x0 + 0.35, -0.15))):
                g.add(Arrow(a, a + (b - a) * max(frac, 0.02), buff=0, color=BLUE, stroke_width=5, max_tip_length_to_length_ratio=0.4).set_opacity(0.25 + 0.75 * frac))
            return g

        def plate_charges():
            q = Vc(clock.get_value()) / V0
            k = int(round(7 * q))
            xs = np.linspace(x1 - 0.45, x1 + 0.45, 7)[:k]
            return VGroup(*[Text("+", font_size=22, color=RED).move_to(_p(x, 0.42)) for x in xs], *[Text("−", font_size=22, color=BLUE).move_to(_p(x, -0.42)) for x in xs])

        readout = always_redraw(
            lambda: VGroup(
                Text(f"t = {clock.get_value():.2f} s", font_size=22),
                Text(f"V_C = {Vc(clock.get_value()):.2f} V", font_size=22, color=YELLOW),
                Text(f"I = {1e3 * I(clock.get_value()):.2f} mA", font_size=22, color=BLUE),
            ).arrange(RIGHT, buff=0.5).move_to(_p(-4.0, -2.75))
        )
        self.add(v_curve, i_curve, readout, always_redraw(current_arrows), always_redraw(plate_charges))
        self.play(clock.animate.set_value(tau), run_time=3.0, rate_func=linear)
        m1 = VGroup(
            DashedLine(axes.c2p(1, 0), axes.c2p(1, 1), color=GREY_B, stroke_width=2),
            Dot(axes.c2p(1, Vc(tau) / V0), color=YELLOW),
            Dot(axes.c2p(1, I(tau) / I0), color=BLUE),
            Text(f"t = RC: V_C = {Vc(tau) / V0:.3f} V0, I = {I(tau) / I0:.3f} I0", font_size=18).next_to(axes.c2p(1, 0.5), RIGHT, buff=0.15),
        )
        self.play(FadeIn(m1))
        self.wait(1.2)
        self.play(clock.animate.set_value(t_end), run_time=6.0, rate_func=linear)
        m5 = Text(f"t = 5 RC: V_C = {Vc(t_end) / V0:.3f} V0", font_size=18).next_to(axes.c2p(5, Vc(t_end) / V0), DOWN, buff=0.3).shift(LEFT * 0.8)
        t = np.linspace(0, 40 * tau, 400001)
        dissipated = np.trapezoid(I(t) ** 2 * R, t)
        stored = 0.5 * C * V0**2
        note = _caption(f"stored in C: ½CV0² = {1e3 * stored:.2f} mJ.   dissipated in R: ∫I²R dt = {1e3 * dissipated:.2f} mJ.   The battery supplied CV0² = {1e3 * C * V0**2:.1f} mJ.")
        self.play(FadeIn(m5), FadeIn(note))
        self.wait(3)


# ---------------------------------------------------------------------------------------------
# PhasorRLC


class PhasorRLC(Scene):
    """Primer M5.0c and notebook M5.3: series RLC with R = 10 Ohm, L = 10 mH, C = 10 uF, source amplitude 1 V.

    omega0 = 1 / sqrt(LC) = 3162 rad/s and Q = sqrt(L/C) / R = 3.16. Phasors V_R = I R, V_L = j omega L I,
    V_C = I / (j omega C), with I = V / Z. The current at one frequency is checked against emlib.circuits.mna.
    """

    R, L, C, V0 = 10.0, 10e-3, 10e-6, 1.0
    SCALE_ROTATING = 1.15  # screen units per volt, part one
    SCALE_SWEEP = 0.62  # part two, where |V_L| reaches Q V0

    def phasors(self, w):
        Z = self.R + 1j * (w * self.L - 1.0 / (w * self.C))
        I = self.V0 / Z
        return I, I * self.R, I * 1j * w * self.L, I / (1j * w * self.C)

    def check_against_mna(self, w):
        V, Ivs = ci.mna(4, resistors=[(1, 2, self.R)], inductors=[(2, 3, self.L)], capacitors=[(3, 0, self.C)], voltage_sources=[(1, 0, self.V0)], omega=w)
        I_mna = -Ivs[0]  # mna reports the current inside the source from + to -
        I, *_ = self.phasors(w)
        assert np.isclose(I_mna, I), (I_mna, I)

    def arrows(self, origin, parts, colours, theta, scale):
        rot = np.exp(1j * theta)
        g = VGroup()
        tip = 0j
        for z, col in zip(parts, colours):
            a = tip
            b = tip + z * rot
            g.add(Arrow(origin + scale * _p(a.real, a.imag), origin + scale * _p(b.real, b.imag), buff=0, color=col, stroke_width=5, max_tip_length_to_length_ratio=0.18, max_stroke_width_to_length_ratio=12))
            tip = b
        return g

    def construct(self):
        w0 = ci.resonance_omega(self.L, self.C)
        Q = ci.quality_factor(self.R, self.L, self.C)
        self.check_against_mna(0.8 * w0)
        colours = {"R": GREEN, "L": ORANGE, "C": PURPLE_B, "S": YELLOW}

        # part 1: rotating phasors at omega = 0.8 omega0 and their projections
        w = 0.8 * w0
        I, VR, VL, VC = self.phasors(w)
        title = _title("Series RLC: phasors add head to tail to the source voltage")
        origin = _p(-4.4, -0.2)
        axes = _plain_axes([0, 4 * np.pi, np.pi], [-2.6, 2.6, 1], 6.4, 3.6).move_to(_p(2.9, -0.2))
        h_axis = Line(origin + _p(-2.1, 0), origin + _p(2.1, 0), color=GREY_D)
        v_axis = Line(origin + _p(0, -2.1), origin + _p(0, 2.1), color=GREY_D)
        re_label = Text("real axis", font_size=18, color=GREY_B).next_to(h_axis, DOWN, buff=0.1).align_to(h_axis, RIGHT)
        legend = VGroup(
            Text(f"V_R = {abs(VR):.2f} V", font_size=20, color=colours["R"]),
            Text(f"V_L = {abs(VL):.2f} V", font_size=20, color=colours["L"]),
            Text(f"V_C = {abs(VC):.2f} V", font_size=20, color=colours["C"]),
            Text(f"source {self.V0:.2f} V", font_size=20, color=colours["S"]),
        ).arrange(RIGHT, buff=0.45).next_to(title, DOWN, buff=0.2)
        info = Text(f"ω = 0.8 ω0, below resonance: V_C > V_L, the current leads the source by {np.degrees(np.angle(I)):.0f}°", font_size=20, color=GREY_B)
        _fit(info).to_edge(DOWN, buff=0.35)
        time_label = Text("ωt", font_size=18, color=GREY_B).next_to(axes.x_axis, DOWN, buff=0.1).align_to(axes.x_axis, RIGHT)
        self.play(FadeIn(title), FadeIn(legend), Create(h_axis), Create(v_axis), FadeIn(re_label), Create(axes), FadeIn(time_label), FadeIn(info))

        theta = ValueTracker(0.0)
        diagram = always_redraw(lambda: VGroup(self.arrows(origin, [VR, VL, VC], [colours["R"], colours["L"], colours["C"]], theta.get_value(), self.SCALE_ROTATING), self.arrows(origin, [self.V0 + 0j], [colours["S"]], theta.get_value(), self.SCALE_ROTATING)))

        def trace(z, col):
            return always_redraw(
                lambda: axes.plot(lambda s: (z * np.exp(1j * s)).real, x_range=[0, max(theta.get_value(), 1e-3)], color=col, stroke_width=3)
            )

        traces = VGroup(trace(VR, colours["R"]), trace(VL, colours["L"]), trace(VC, colours["C"]), trace(self.V0 + 0j, colours["S"]))
        self.add(diagram, traces)
        self.play(theta.animate.set_value(4 * np.pi), run_time=9, rate_func=linear)
        note = Text("projections on the real axis are the voltages you would measure; at every instant V_R + V_L + V_C = source", font_size=18)
        _fit(note).to_edge(DOWN, buff=0.35)
        self.play(Transform(info, note))
        self.wait(1.5)

        # part 2: sweep the frequency with the current along the real axis
        self.play(FadeOut(traces), FadeOut(diagram), FadeOut(axes), FadeOut(time_label), FadeOut(legend), FadeOut(info))
        self.remove(diagram, traces)
        title2 = _title("Sweeping through resonance: V_L and V_C cancel at ω0 and the current peaks")
        self.play(Transform(title, title2))
        ratio = ValueTracker(0.5)
        origin2 = _p(-4.2, -1.1)
        h2 = Line(origin2 + _p(-1.2, 0), origin2 + _p(2.4, 0), color=GREY_D)
        v2 = Line(origin2 + _p(0, -1.4), origin2 + _p(0, 2.6), color=GREY_D)
        self.play(Transform(h_axis, h2), Transform(v_axis, v2), FadeOut(re_label))

        def frozen():
            I, VR, VL, VC = self.phasors(ratio.get_value() * w0)
            rot = np.exp(-1j * np.angle(I))  # current along the real axis
            return VGroup(
                self.arrows(origin2, [VR * rot, VL * rot, VC * rot], [colours["R"], colours["L"], colours["C"]], 0.0, self.SCALE_SWEEP),
                self.arrows(origin2, [self.V0 * rot], [colours["S"]], 0.0, self.SCALE_SWEEP),
            )

        rs = np.geomspace(0.5, 2.0, 400)
        Iabs = np.array([abs(self.phasors(r * w0)[0]) for r in rs])
        ax2 = _plain_axes([np.log10(0.5), np.log10(2.0), 0.30103], [0, 110, 50], 5.2, 3.2).move_to(_p(3.2, -0.3))
        ax2_lab = VGroup(
            Text("|I| (mA)", font_size=18, color=GREY_B).next_to(ax2.y_axis, UP, buff=0.1),
            Text("ω/ω0 (log scale)", font_size=18, color=GREY_B).next_to(ax2.x_axis, DOWN, buff=0.35).align_to(ax2.x_axis, RIGHT),
            *[Text(t, font_size=16, color=GREY_B).next_to(ax2.c2p(np.log10(v), 0), DOWN, buff=0.1) for t, v in (("0.5", 0.5), ("1", 1.0), ("2", 2.0))],
            Text("100", font_size=16, color=GREY_B).next_to(ax2.c2p(np.log10(0.5), 100), LEFT, buff=0.1),
        )
        curve = always_redraw(
            lambda: ax2.plot(lambda lx: 1e3 * abs(self.phasors(10**lx * w0)[0]), x_range=[np.log10(0.5), max(np.log10(ratio.get_value()), np.log10(0.5) + 1e-3)], color=BLUE, stroke_width=4)
        )
        readout = always_redraw(
            lambda: VGroup(
                Text(f"ω/ω0 = {ratio.get_value():.2f}", font_size=22),
                Text(f"|I| = {1e3 * abs(self.phasors(ratio.get_value() * w0)[0]):5.1f} mA", font_size=22, color=BLUE),
                Text(f"|V_L| = {abs(self.phasors(ratio.get_value() * w0)[2]):.2f} V", font_size=22, color=colours["L"]),
                Text(f"|V_C| = {abs(self.phasors(ratio.get_value() * w0)[3]):.2f} V", font_size=22, color=colours["C"]),
            ).arrange(RIGHT, buff=0.45).to_edge(DOWN, buff=0.35)
        )
        diagram2 = always_redraw(frozen)
        self.play(Create(ax2), FadeIn(ax2_lab))
        self.add(diagram2, curve, readout)
        self.play(ratio.animate.set_value(1.0), run_time=5, rate_func=linear)
        res_note = Text(f"at ω0: |V_L| = |V_C| = Q V0 = {Q:.2f} V, they cancel; |I| = V0/R = {1e3 * self.V0 / self.R:.0f} mA", font_size=20, color=YELLOW)
        _fit(res_note).next_to(readout, UP, buff=0.2)
        self.play(FadeIn(res_note))
        self.wait(2)
        self.play(FadeOut(res_note))
        self.play(ratio.animate.set_value(2.0), run_time=5, rate_func=linear)
        self.wait(2)


# ---------------------------------------------------------------------------------------------
# PercolationBackbone


class PercolationBackbone(Scene):
    """Primer M5.0d and notebook M5.4: a 20 by 20 bond lattice with one fixed random number per bond.

    A bond is present when its number is below p, so bonds only appear as p rises. Each state is solved
    with emlib.percolation (left column at V = 1, right at V = 0).
    """

    N = 20
    SPACING = 0.27
    SEED = 7

    def states(self, ps):
        rng = np.random.default_rng(self.SEED)
        uh, uv = rng.random((self.N - 1, self.N)), rng.random((self.N, self.N - 1))
        out = []
        for p in ps:
            h, v = uh < p, uv < p
            _, Ih, Iv, _ = pc.solve_lattice(h, v)
            out.append((h, v, np.abs(Ih), np.abs(Iv), pc.effective_conductivity(h, v), pc.spans(h, v)))
        return out

    def construct(self):
        n, s = self.N, self.SPACING
        ps = np.round(np.linspace(0.30, 0.70, 81), 3)
        states = self.states(ps)
        span_idx = next(k for k, st in enumerate(states) if st[5])
        origin = _p(-5.3, -2.4)  # node (0, 0)

        def node(i, j):
            return origin + _p(i * s, j * s)

        title = _title("Percolation: current flows once a cluster spans the lattice")
        lines_h = [[Line(node(i, j), node(i + 1, j), stroke_width=3) for j in range(n)] for i in range(n - 1)]
        lines_v = [[Line(node(i, j), node(i, j + 1), stroke_width=3) for j in range(n - 1)] for i in range(n)]
        all_lines = VGroup(*[l for row in lines_h for l in row], *[l for row in lines_v for l in row])
        left = Line(node(0, 0) + _p(-0.12, -0.1), node(0, n - 1) + _p(-0.12, 0.1), color=RED, stroke_width=8)
        right = Line(node(n - 1, 0) + _p(0.12, -0.1), node(n - 1, n - 1) + _p(0.12, 0.1), color=BLUE, stroke_width=8)
        lab_l = Text("V = 1", font_size=20, color=RED).next_to(left, UP, buff=0.12)
        lab_r = Text("V = 0", font_size=20, color=BLUE).next_to(right, UP, buff=0.12)

        k_tracker = ValueTracker(0)

        def style(_mob):
            h, v, Ih, Iv, *_ = states[int(round(k_tracker.get_value()))]
            imax = max(Ih.max(initial=0.0), Iv.max(initial=0.0))
            for present, cur, rows in ((h, Ih, lines_h), (v, Iv, lines_v)):
                for i, row in enumerate(rows):
                    for j, ln in enumerate(row):
                        if not present[i, j]:
                            ln.set_stroke(GREY_D, width=1, opacity=0.25)
                        elif imax > 0 and cur[i, j] > 1e-9 * imax:
                            f = np.sqrt(cur[i, j] / imax)
                            ln.set_stroke(interpolate_color(ORANGE, YELLOW, f), width=2.5 + 4 * f, opacity=1.0)
                        else:
                            ln.set_stroke(GREY_B, width=2.5, opacity=0.7)

        all_lines.add_updater(style)
        style(all_lines)

        ax = _plain_axes([0.3, 0.7, 0.1], [0, 0.45, 0.1], 4.2, 2.6).move_to(_p(3.6, -0.9))
        ax_lab = VGroup(
            Text("σ_eff", font_size=20, color=GREY_B).next_to(ax.y_axis, UP, buff=0.1),
            Text("p", font_size=20, color=GREY_B).next_to(ax.c2p(0.7, 0), UP + RIGHT, buff=0.1),
            *[Text(f"{v:.1f}", font_size=16, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0.3, 0.5, 0.7)],
            Text("0.4", font_size=16, color=GREY_B).next_to(ax.c2p(0.3, 0.4), LEFT, buff=0.1),
        )
        pc_line = DashedLine(ax.c2p(0.5, 0), ax.c2p(0.5, 0.45), color=GREY_B, stroke_width=2)
        pc_lab = Text("p_c = 1/2", font_size=16, color=GREY_B).next_to(ax.c2p(0.5, 0.45), RIGHT, buff=0.08)
        sig = np.array([st[4] for st in states])
        curve = always_redraw(
            lambda: VMobject(stroke_color=YELLOW, stroke_width=3).set_points_as_corners([ax.c2p(p, v) for p, v in zip(ps[: int(round(k_tracker.get_value())) + 1], sig[: int(round(k_tracker.get_value())) + 1])])
            if k_tracker.get_value() >= 1
            else VMobject()
        )

        def readout_mob():
            k = int(round(k_tracker.get_value()))
            h, v, Ih, Iv, sg, sp = states[k]
            bb_h, bb_v = pc.backbone(*pc.solve_lattice(h, v)[1:3]) if sp else (np.zeros_like(h), np.zeros_like(v))
            return VGroup(
                Text(f"p = {ps[k]:.3f}", font_size=24),
                Text(f"bonds present: {int(h.sum() + v.sum())} of {h.size + v.size}", font_size=20, color=GREY_B),
                Text(f"spanning cluster: {'yes' if sp else 'no'}", font_size=20, color=YELLOW if sp else GREY_B),
                Text(f"current-carrying bonds: {int(bb_h.sum() + bb_v.sum())}", font_size=20, color=ORANGE),
                Text(f"σ_eff = {sg:.3f}", font_size=24, color=YELLOW),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(_p(3.3, 1.75))

        readout = always_redraw(readout_mob)
        legend = _caption("grey: bonds with no current (isolated clusters and dangling ends).  orange to yellow: current, brighter is larger", 18)
        self.play(FadeIn(title), FadeIn(all_lines), FadeIn(left), FadeIn(right), FadeIn(lab_l), FadeIn(lab_r), Create(ax), FadeIn(ax_lab), Create(pc_line), FadeIn(pc_lab), FadeIn(legend))
        self.add(readout, curve)
        self.play(k_tracker.animate.set_value(span_idx), run_time=7, rate_func=linear)
        note = _caption(f"first spanning cluster at p = {ps[span_idx]:.3f}: all the current runs on its backbone, the dangling ends carry none", 18)
        self.play(Transform(legend, note))
        self.wait(2.5)
        self.play(k_tracker.animate.set_value(len(ps) - 1), run_time=6, rate_func=linear)
        end = _caption(f"p = 0.70: σ_eff = {sig[-1]:.3f}; effective-medium estimate 2(p − 1/2) = {2 * (0.70 - 0.5):.2f}", 18)
        self.play(Transform(legend, end))
        self.wait(2.5)
