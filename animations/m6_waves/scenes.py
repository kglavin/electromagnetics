"""Module 6 animations: induction, inductance, electromagnetic waves and the Yee grid.

FaradayInduction  primer 00a and notebook 01: a magnetic dipole moves along the axis of a loop; the flux
                  through the loop and the emf are plotted live, and the induced current flips as the
                  magnet passes the centre (Lenz).
CoupledCoils      primer 00b and notebook 02: the current in loop 1 oscillates; its field threads loop 2,
                  which carries an induced current opposing the change; then loop 2 moves away and M falls.
PlaneWave         primer 00c: a plane wave along x with E along y and B along z, in phase, with the
                  Poynting vector along x and a running count of the energy through a plane.
YeeLeapfrog       primer 00d and notebook 03: E on nodes and H on half nodes updated in turn; a pulse
                  moves right; then the same grid at Courant number 1.02 blows up.

Render with:
  uv run manim -ql --media_dir media/m6_waves animations/m6_waves/scenes.py FaradayInduction
  uv run manim -ql --media_dir media/m6_waves animations/m6_waves/scenes.py CoupledCoils
  uv run manim -ql --media_dir media/m6_waves animations/m6_waves/scenes.py PlaneWave
  uv run manim -ql --media_dir media/m6_waves animations/m6_waves/scenes.py YeeLeapfrog
"""

import numpy as np
from manim import (
    BLUE,
    BLUE_B,
    DEGREES,
    DOWN,
    GREY,
    GREY_B,
    GREY_D,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Arrow3D,
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    Ellipse,
    FadeIn,
    FadeOut,
    Line,
    Polygon,
    Rectangle,
    Scene,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

from emlib import biot_savart as bs
from emlib import fdtd as fd
from emlib import induction as ind
from emlib.fields import EPS0

MU0 = bs.MU0


def _p(x, y, z=0.0):
    return np.array([x, y, z])


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


def _current_marker(centre, out_of_page, size=0.13, colour=ORANGE, opacity=1.0):
    """A dot in a circle (current out of the page) or a cross in a circle (into the page)."""
    ring = Circle(radius=size, stroke_color=colour, stroke_width=2, stroke_opacity=opacity, fill_opacity=0).move_to(centre)
    if out_of_page:
        mark = Dot(centre, radius=size * 0.32, color=colour, fill_opacity=opacity, stroke_width=0)
    else:
        d = size * 0.55
        mark = VGroup(
            Line(centre + _p(-d, -d), centre + _p(d, d), color=colour, stroke_width=2.5, stroke_opacity=opacity),
            Line(centre + _p(-d, d), centre + _p(d, -d), color=colour, stroke_width=2.5, stroke_opacity=opacity),
        )
    return VGroup(ring, mark)


# ---------------------------------------------------------------------------------------------
# FaradayInduction


class FaradayInduction(Scene):
    """A dipole m along +x passes through a loop of radius R at speed v.

    Flux through the loop with the dipole at axial distance z: Phi = mu0 m R^2 / (2 (R^2 + z^2)^(3/2)).
    emf = -dPhi/dt = -v dPhi/dz. Positive emf drives current counter-clockwise seen from +x, which
    at the top of the edge-on loop points out of the page.
    """

    R, M, V = 0.10, 1.0, 1.0  # m, A m^2, m/s
    Z0, Z1 = -0.40, 0.40  # m
    SCALE = 6.0  # screen units per metre
    LOOP_X, AXIS_Y = -3.6, 0.2

    def flux(self, z):
        return MU0 * self.M * self.R**2 / (2.0 * (self.R**2 + z**2) ** 1.5)

    def emf(self, z):
        return -self.V * MU0 * self.M * self.R**2 * (-3.0 * z) / (2.0 * (self.R**2 + z**2) ** 2.5)

    def check_flux(self):
        z = 0.05
        num = ind.disk_flux(lambda p: bs.dipole_B([self.M, 0, 0], p - np.array([-z, 0, 0])), (0, 0, 0), (1, 0, 0), self.R)
        assert abs(num / self.flux(z) - 1.0) < 2e-3, (num, self.flux(z))
        return num

    def construct(self):
        self.check_flux()
        title = _title("Faraday's law: a changing flux drives an emf around the loop")
        self.add(title)

        z_of = lambda t: self.Z0 + self.V * t
        T = (self.Z1 - self.Z0) / self.V
        tk = ValueTracker(0.0)
        x_of = lambda z: self.LOOP_X + self.SCALE * z

        axis = DashedLine(_p(-6.9, self.AXIS_Y), _p(-0.4, self.AXIS_Y), color=GREY_D, stroke_width=1.5)
        loop = Ellipse(width=0.35, height=2 * self.R * self.SCALE, color=ORANGE, stroke_width=4).move_to(_p(self.LOOP_X, self.AXIS_Y))
        loop_label = Text("loop, R = 10 cm", font_size=18, color=ORANGE).next_to(loop, DOWN, buff=0.6)
        self.add(axis, loop, loop_label)

        # dipole field lines r = r0 sin^2(theta), in screen units, moved with the magnet
        lines = VGroup()
        th = np.linspace(0.02, np.pi - 0.02, 160)
        for r0 in (0.8, 1.4, 2.1):
            r = r0 * np.sin(th) ** 2
            for s in (1, -1):
                pts = np.column_stack([r * np.cos(th), s * r * np.sin(th), np.zeros_like(th)])
                vm = VMobject(stroke_color=BLUE_B, stroke_width=1.5, stroke_opacity=0.55)
                vm.set_points_smoothly(pts)
                lines.add(vm)
        magnet = VGroup(
            Rectangle(width=0.45, height=0.28, color=BLUE, fill_color=BLUE, fill_opacity=0.9, stroke_width=0).shift(LEFT * 0.225),
            Rectangle(width=0.45, height=0.28, color=RED, fill_color=RED, fill_opacity=0.9, stroke_width=0).shift(RIGHT * 0.225),
        )
        nlab = Text("N", font_size=16, color=WHITE).move_to(magnet[1])
        slab = Text("S", font_size=16, color=WHITE).move_to(magnet[0])
        mag = VGroup(lines, magnet, nlab, slab)
        mag.add_updater(lambda m: m.shift(_p(x_of(z_of(tk.get_value())), self.AXIS_Y) - magnet.get_center()))
        self.add(mag)

        # live plots
        flux_max, emf_max = self.flux(0.0), np.max(np.abs(self.emf(np.linspace(self.Z0, self.Z1, 2001))))
        ax_f = _plain_axes([0, T, T / 4], [0, 1.1 * flux_max * 1e6, 2], 5.4, 1.9).move_to(_p(3.7, 1.55))
        ax_e = _plain_axes([0, T, T / 4], [-1.15 * emf_max * 1e6, 1.15 * emf_max * 1e6, 20], 5.4, 1.9).move_to(_p(3.7, -1.15))
        lab_f = Text("flux Φ through the loop (μWb)", font_size=17, color=BLUE_B).next_to(ax_f, UP, buff=0.08)
        lab_e = Text("emf = −dΦ/dt (μV)", font_size=17, color=YELLOW).next_to(ax_e, UP, buff=0.08)
        tlab = Text("time", font_size=15, color=GREY_B).next_to(ax_e, DOWN, buff=0.08).align_to(ax_e, RIGHT)
        self.add(ax_f, ax_e, lab_f, lab_e, tlab)

        def curve(ax, fn, colour):
            t = tk.get_value()
            if t < 1e-3:
                return VMobject()
            return ax.plot(fn, x_range=[0, t, t / 200], color=colour, stroke_width=3)

        f_curve = always_redraw(lambda: curve(ax_f, lambda t: self.flux(z_of(t)) * 1e6, BLUE_B))
        e_curve = always_redraw(lambda: curve(ax_e, lambda t: self.emf(z_of(t)) * 1e6, YELLOW))
        self.add(f_curve, e_curve)

        def readout():
            z = z_of(tk.get_value())
            e = self.emf(z)
            if abs(e) < 0.02 * emf_max:
                state = "magnet at the loop: Φ at its peak, emf passes through zero"
            elif e < 0:
                state = "approaching: Φ rising, the induced current's field opposes the rise"
            else:
                state = "leaving: Φ falling, the induced current tries to keep it"
            t1 = Text(f"z = {z * 100:+5.1f} cm    Φ = {self.flux(z) * 1e6:5.2f} μWb    emf = {e * 1e6:+6.1f} μV", font_size=19)
            t2 = Text(state, font_size=19, color=GREY_B)
            return VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN, buff=0.35).to_edge(LEFT, buff=0.4)

        info = always_redraw(readout)
        self.add(info)

        def marker():
            e = self.emf(z_of(tk.get_value()))
            op = min(1.0, abs(e) / emf_max)
            top = _p(self.LOOP_X, self.AXIS_Y + self.R * self.SCALE)
            bot = _p(self.LOOP_X, self.AXIS_Y - self.R * self.SCALE)
            out_top = e > 0
            return VGroup(
                _current_marker(top + UP * 0.25, out_top, opacity=op),
                _current_marker(bot + DOWN * 0.25, not out_top, opacity=op),
            )

        mk = always_redraw(marker)
        self.add(mk)
        legend = Text("induced current: ⊙ out of the page, ⊗ into it", font_size=15, color=ORANGE).move_to(_p(-3.6, 2.85))
        mlab = Text("dipole m = 1 A m², moving at 1 m/s", font_size=15, color=GREY_B).move_to(_p(-3.6, 2.5))
        self.add(legend, mlab)

        self.play(tk.animate.set_value(T), run_time=12, rate_func=linear)
        self.wait(1.5)


# ---------------------------------------------------------------------------------------------
# CoupledCoils


def _loop_field_lines(R, I, starts, box, step=0.004, n_max=1500):
    """Field lines of a loop of radius R with axis along x, in the x-y meridian plane, traced both ways."""
    mid, dl = bs.segments(bs.loop(R, 120, normal=(1, 0, 0)))
    B = lambda p: bs.wire_B(mid, dl, I, p[None, :])[0]

    def trace(p0, sign):
        pts = [p0.copy()]
        p = p0.copy()
        for k in range(n_max):
            def f(q):
                b = B(q)
                return sign * b / np.linalg.norm(b)
            k1 = f(p)
            k2 = f(p + 0.5 * step * k1)
            k3 = f(p + 0.5 * step * k2)
            k4 = f(p + step * k3)
            p = p + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            pts.append(p.copy())
            if not (box[0] < p[0] < box[1] and box[2] < p[1] < box[3]):
                break
            if k > 20 and np.linalg.norm(p - p0) < 1.5 * step:
                break
        return np.array(pts)

    lines = []
    for y0 in starts:
        p0 = np.array([0.0, y0, 0.0])
        fwd, bwd = trace(p0, 1.0), trace(p0, -1.0)
        lines.append(np.vstack([bwd[::-1], fwd[1:]]))
    return lines


class CoupledCoils(Scene):
    """Two coaxial loops of radius 10 cm. I1 = I0 sin(2 pi f t); emf2 = -M dI1/dt; M from the exact formula."""

    A = B_R = 0.10
    I0, F = 1.0, 50.0
    SCALE = 9.0
    X1, AXIS_Y = -5.6, 0.25

    def construct(self):
        title = _title("Mutual inductance: the current in one loop drives an emf in the other")
        self.add(title)
        tk = ValueTracker(0.0)  # physical time, s
        dk = ValueTracker(0.05)  # separation, m
        w = 2 * np.pi * self.F
        I1 = lambda: self.I0 * np.sin(w * tk.get_value())
        dI1 = lambda: self.I0 * w * np.cos(w * tk.get_value())
        M = lambda: ind.coaxial_loops_M(self.A, self.B_R, dk.get_value())

        # field lines of loop 1 (precomputed), opacity follows |I1|
        x_lo, x_hi = (-6.9 - self.X1) / self.SCALE, (0.3 - self.X1) / self.SCALE
        raw = _loop_field_lines(self.A, 1.0, [0.02, 0.05, 0.075, 0.09, -0.02, -0.05, -0.075, -0.09], (x_lo, x_hi, -0.3, 0.3))
        lines = VGroup()
        for pts in raw:
            s = np.column_stack([self.X1 + self.SCALE * pts[:, 0], self.AXIS_Y + self.SCALE * pts[:, 1], np.zeros(len(pts))])
            vm = VMobject(stroke_width=1.6)
            vm.set_points_as_corners(s)
            lines.add(vm)

        def colour_lines(m):
            i = I1()
            m.set_stroke(color=YELLOW if i >= 0 else BLUE_B, opacity=0.15 + 0.7 * abs(i) / self.I0)

        lines.add_updater(colour_lines)
        axis = DashedLine(_p(-6.9, self.AXIS_Y), _p(0.3, self.AXIS_Y), color=GREY_D, stroke_width=1.5)
        h = 2 * self.A * self.SCALE
        loop1 = Ellipse(width=0.35, height=h, color=ORANGE, stroke_width=4).move_to(_p(self.X1, self.AXIS_Y))
        loop2 = always_redraw(lambda: Ellipse(width=0.35, height=h, color=GREY_B, stroke_width=4).move_to(_p(self.X1 + self.SCALE * dk.get_value(), self.AXIS_Y)))
        l1 = Text("loop 1: I₁ = I₀ sin ωt", font_size=16, color=ORANGE).move_to(_p(self.X1 - 0.2, self.AXIS_Y - h / 2 - 0.7))
        l2 = always_redraw(lambda: Text("loop 2", font_size=16, color=GREY_B).move_to(_p(self.X1 + self.SCALE * dk.get_value() + 0.3, self.AXIS_Y - h / 2 - 1.05)))
        self.add(lines, axis, loop1, loop2, l1, l2)

        def markers():
            i = I1()
            e2 = -M() * dI1()
            emax = M() * self.I0 * w
            top1 = _p(self.X1, self.AXIS_Y + h / 2 + 0.25)
            bot1 = _p(self.X1, self.AXIS_Y - h / 2 - 0.25)
            x2 = self.X1 + self.SCALE * dk.get_value()
            top2 = _p(x2, self.AXIS_Y + h / 2 + 0.25)
            bot2 = _p(x2, self.AXIS_Y - h / 2 - 0.25)
            op1 = abs(i) / self.I0
            op2 = min(1.0, abs(e2) / max(emax, 1e-30))
            return VGroup(
                _current_marker(top1, i > 0, colour=ORANGE, opacity=op1),
                _current_marker(bot1, i <= 0, colour=ORANGE, opacity=op1),
                _current_marker(top2, e2 > 0, colour=GREY_B, opacity=op2),
                _current_marker(bot2, e2 <= 0, colour=GREY_B, opacity=op2),
            )

        self.add(always_redraw(markers))

        def readout():
            e2 = -M() * dI1()
            lines_txt = [
                Text(f"d = {dk.get_value() * 100:4.1f} cm", font_size=20),
                Text(f"M = {M() * 1e9:6.2f} nH", font_size=20, color=YELLOW),
                Text(f"I₁ = {I1():+5.2f} A   (f = 50 Hz)", font_size=20, color=ORANGE),
                Text(f"emf₂ = −M dI₁/dt = {e2 * 1e6:+6.1f} μV", font_size=20, color=GREY_B),
            ]
            return VGroup(*lines_txt).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to(_p(4.1, 1.6))

        self.add(always_redraw(readout))
        note = Text("loop 2's current opposes the change of the flux from loop 1 (Lenz)", font_size=17, color=GREY_B)
        note = _fit(note).to_edge(DOWN, buff=0.3)
        self.add(note)

        # M against separation
        ds = np.linspace(0.02, 0.45, 200)
        Ms = ind.coaxial_loops_M(self.A, self.B_R, ds) * 1e9
        ax = _plain_axes([0, 0.45, 0.1], [0, 1.1 * Ms.max(), 50], 4.6, 2.0).move_to(_p(4.1, -1.35))
        ax_lab = Text("M (nH) against separation d", font_size=16, color=YELLOW).next_to(ax, UP, buff=0.08)
        d_lab = Text("d (m): 0 to 0.45", font_size=14, color=GREY_B).next_to(ax, DOWN, buff=0.08).align_to(ax, RIGHT)
        curve = ax.plot(lambda d: ind.coaxial_loops_M(self.A, self.B_R, max(d, 0.02)) * 1e9, x_range=[0.02, 0.45, 0.005], color=YELLOW, stroke_width=2.5)
        dot = always_redraw(lambda: Dot(ax.c2p(dk.get_value(), M() * 1e9), color=WHITE, radius=0.07))
        self.add(ax, ax_lab, d_lab, curve, dot)

        self.play(tk.animate.set_value(3 / self.F), run_time=9, rate_func=linear)
        self.play(tk.animate.set_value(7 / self.F), dk.animate.set_value(0.40), run_time=10, rate_func=linear)
        self.wait(1)



# ---------------------------------------------------------------------------------------------
# PlaneWave


class PlaneWave(ThreeDScene):
    """E = E0 cos(kx - wt) y_hat, B = (E0 / c) cos(kx - wt) z_hat, S = E x B / mu0 along x.

    Physical values for the readout: E0 = 100 V/m, f = 1 GHz. The time average of S is E0^2 / (2 eta0).
    The energy through 1 m^2 of a plane up to time t is (E0^2 / eta0) (t/2 + sin(2 w t) / (4 w)) for the
    phase at that plane.
    """

    E0, F = 100.0, 1e9
    N = 26
    XS = 5.2  # half-length of the drawn axis, screen units
    WAVELENGTHS = 2.0  # on screen
    AMP = 1.6  # screen amplitude of E and B arrows

    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-62 * DEGREES, zoom=0.9)
        tk = ValueTracker(0.0)  # phase time in periods
        kx = 2 * np.pi * self.WAVELENGTHS / (2 * self.XS)
        xs = np.linspace(-self.XS, self.XS, self.N)
        phase = lambda x: kx * x - 2 * np.pi * tk.get_value()

        x_axis = Line(_p(-self.XS - 0.3, 0, 0), _p(self.XS + 0.6, 0, 0), color=GREY, stroke_width=2)
        y_axis = Line(_p(-self.XS, -2.0, 0), _p(-self.XS, 2.0, 0), color=GREY_D, stroke_width=1)
        z_axis = Line(_p(-self.XS, 0, -2.0), _p(-self.XS, 0, 2.0), color=GREY_D, stroke_width=1)
        self.add(x_axis, y_axis, z_axis)

        e_lines = VGroup(*[Line(_p(x, 0, 0), _p(x, 0.01, 0), color=YELLOW, stroke_width=3) for x in xs])
        b_lines = VGroup(*[Line(_p(x, 0, 0), _p(x, 0, 0.01), color=BLUE_B, stroke_width=3) for x in xs])

        def upd_e(g):
            for x, ln in zip(xs, g):
                ln.put_start_and_end_on(_p(x, 0, 0), _p(x, self.AMP * np.cos(phase(x)) + 1e-4, 0))

        def upd_b(g):
            for x, ln in zip(xs, g):
                ln.put_start_and_end_on(_p(x, 0, 0), _p(x, 0, self.AMP * np.cos(phase(x)) + 1e-4))

        e_lines.add_updater(upd_e)
        b_lines.add_updater(upd_b)
        xf = np.linspace(-self.XS, self.XS, 160)
        e_env = VMobject(stroke_color=YELLOW, stroke_width=2)
        b_env = VMobject(stroke_color=BLUE_B, stroke_width=2)
        e_env.add_updater(lambda m: m.set_points_smoothly(np.column_stack([xf, self.AMP * np.cos(phase(xf)), 0 * xf])))
        b_env.add_updater(lambda m: m.set_points_smoothly(np.column_stack([xf, 0 * xf, self.AMP * np.cos(phase(xf))])))
        upd_e(e_lines), upd_b(b_lines)
        e_env.set_points_smoothly(np.column_stack([xf, self.AMP * np.cos(phase(xf)), 0 * xf]))
        b_env.set_points_smoothly(np.column_stack([xf, 0 * xf, self.AMP * np.cos(phase(xf))]))
        self.add(e_lines, b_lines, e_env, b_env)

        S_arrow = Arrow3D(_p(self.XS + 0.2, 0, 0), _p(self.XS + 1.6, 0, 0), color=RED, resolution=8)
        self.add(S_arrow)
        xp = 2.0
        plane = Polygon(_p(xp, -1.9, -1.9), _p(xp, 1.9, -1.9), _p(xp, 1.9, 1.9), _p(xp, -1.9, 1.9),
                        stroke_color=GREY_B, stroke_width=1, fill_color=GREY, fill_opacity=0.12)
        self.add(plane)

        c, eta = fd.C0, fd.ETA0
        B0 = self.E0 / c
        S_avg = self.E0**2 / (2 * eta)
        w = 2 * np.pi * self.F

        title = Text("A plane wave: E along y, B along z, energy flows along x", font_size=26).to_edge(UP, buff=0.25)
        legend = VGroup(
            Text("E (yellow): E₀ = 100 V/m", font_size=17, color=YELLOW),
            Text(f"B (blue): B₀ = E₀ / c = {B0 * 1e9:.0f} nT", font_size=17, color=BLUE_B),
            Text("S = E × B / μ0 (red), along the direction of travel", font_size=17, color=RED),
            Text(f"E / B = c = {c:.4e} m/s,  E ⊥ B ⊥ x", font_size=17),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UP + LEFT, buff=0.3).shift(DOWN * 0.55)
        self.add_fixed_in_frame_mobjects(title, legend)

        def energy_text():
            t = tk.get_value() / self.F  # seconds
            ph0 = kx * xp  # phase offset at the plane, screen-unit wave mapped to the same cosine
            # energy per m^2 through the plane: integral of (E0^2/eta) cos^2(ph0 - w t) dt
            U = self.E0**2 / eta * (t / 2 - (np.sin(2 * (ph0 - w * t)) - np.sin(2 * ph0)) / (4 * w))
            g = VGroup(
                Text(f"f = 1 GHz,  t = {t * 1e9:4.2f} ns", font_size=18),
                Text(f"energy through 1 m² of the grey plane: {U * 1e9:5.2f} nJ", font_size=18, color=RED),
                Text(f"average power per area E₀²/(2η0) = {S_avg:.2f} W/m²", font_size=18, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(DOWN + LEFT, buff=0.35)
            return g

        info = energy_text()
        self.add_fixed_in_frame_mobjects(info)

        def refresh(m):
            m.become(energy_text())
            self.add_fixed_in_frame_mobjects(m)

        info.add_updater(refresh)
        self.begin_ambient_camera_rotation(rate=0.035)
        self.play(tk.animate.set_value(4.0), run_time=16, rate_func=linear)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)


# ---------------------------------------------------------------------------------------------
# YeeLeapfrog


def yee_run(n_cells, courant, n_steps, Ey0, Hz0, dx=1e-3, boundary="mur"):
    """The same leapfrog as emlib.fdtd.fdtd_1d (vacuum), returning Ey and Hz after every half step."""
    dt = courant * dx / fd.C0
    ch = dt / (MU0 * dx)
    ce = dt / (EPS0 * dx)
    k = (fd.C0 * dt - dx) / (fd.C0 * dt + dx)
    Ey, Hz = np.array(Ey0, float), np.array(Hz0, float)
    states = [("start", Ey.copy(), Hz.copy())]
    for _ in range(n_steps):
        Hz = Hz - ch * (Ey[1:] - Ey[:-1])
        states.append(("H", Ey.copy(), Hz.copy()))
        e0, e1, em, em1 = Ey[0], Ey[1], Ey[-1], Ey[-2]
        Ey[1:-1] = Ey[1:-1] - ce * (Hz[1:] - Hz[:-1])
        if boundary == "mur":
            Ey[0] = e1 + k * (Ey[1] - e0)
            Ey[-1] = em1 + k * (Ey[-2] - em)
        else:
            Ey[0] = Ey[-1] = 0.0
        states.append(("E", Ey.copy(), Hz.copy()))
    return states


class YeeLeapfrog(Scene):
    """1D Yee grid, 48 cells. Part one at S = 0.9: a right-moving Gaussian pulse; the E snapshots are
    checked against emlib.fdtd.fdtd_1d. Part two at S = 1.02: a narrow pulse, and the largest |E|
    grows by orders of magnitude within a few dozen steps."""

    N = 48
    X0, X1 = -6.4, 6.4
    E_BASE, H_BASE, AMP = 1.25, -1.05, 0.85

    def setup_pulse(self, courant, width_cells, centre_cells):
        dx = 1e-3
        dt = courant * dx / fd.C0
        x = np.arange(self.N + 1) * dx
        xh = (np.arange(self.N) + 0.5) * dx
        f = lambda s: np.exp(-(((s - centre_cells * dx) / (width_cells * dx)) ** 2))
        return f(x), f(xh + fd.C0 * dt / 2) / fd.ETA0

    def construct(self):
        title = _title("The Yee leapfrog: H and E updated in turn on a staggered grid")
        self.add(title)
        S1, steps1 = 0.9, 34
        Ey0, Hz0 = self.setup_pulse(S1, 3.0, 9.0)
        states = yee_run(self.N, S1, steps1, Ey0, Hz0)
        ref = fd.fdtd_1d(self.N, 1e-3, steps1, courant=S1, Ey0=Ey0, Hz0=Hz0, snapshot_every=1)
        assert np.allclose(states[-1][1], ref["snapshots"][-1], atol=1e-12)

        sx = np.linspace(self.X0, self.X1, self.N + 1)
        sxh = 0.5 * (sx[1:] + sx[:-1])
        e_base = Line(_p(self.X0 - 0.2, self.E_BASE), _p(self.X1 + 0.2, self.E_BASE), color=GREY_D, stroke_width=1)
        h_base = Line(_p(self.X0 - 0.2, self.H_BASE), _p(self.X1 + 0.2, self.H_BASE), color=GREY_D, stroke_width=1)
        e_lab = Text("Ey on nodes x_i", font_size=17, color=YELLOW).move_to(_p(-5.3, self.E_BASE + 1.15))
        h_lab = Text("η0 Hz on half nodes x_i+1/2", font_size=17, color=BLUE_B).move_to(_p(-4.7, self.H_BASE + 1.15))
        self.add(e_base, h_base, e_lab, h_lab)

        def bars(xs, base, vals, colour, cap=0.95):
            g = VGroup()
            for x, v in zip(xs, vals):
                h = float(np.clip(v * self.AMP, -cap, cap))
                g.add(Line(_p(x, base), _p(x, base + h + (1e-3 if h >= 0 else -1e-3)), color=colour, stroke_width=4))
            return g

        e_bars = bars(sx, self.E_BASE, states[0][1], YELLOW)
        h_bars = bars(sxh, self.H_BASE, states[0][2] * fd.ETA0, BLUE_B)
        self.add(e_bars, h_bars)
        status = Text(f"Courant number S = c Δt / Δx = {S1}: stable (S ≤ 1)", font_size=20).to_edge(DOWN, buff=0.75)
        step_txt = Text("step 0", font_size=20, color=GREY_B).to_edge(DOWN, buff=0.35)
        self.add(status, step_txt)

        def show(kind, Ey, Hz, n, rt):
            new_e = bars(sx, self.E_BASE, Ey, YELLOW if kind != "E" else WHITE)
            new_h = bars(sxh, self.H_BASE, Hz * fd.ETA0, BLUE_B if kind != "H" else WHITE)
            new_step = Text(f"step {n}: {'H updated from the curl of E' if kind == 'H' else 'E updated from the curl of H'}",
                            font_size=20, color=GREY_B).to_edge(DOWN, buff=0.35)
            self.play(e_bars.animate.become(new_e), h_bars.animate.become(new_h), step_txt.animate.become(new_step),
                      run_time=rt, rate_func=linear)

        for idx in range(1, len(states)):
            kind, Ey, Hz = states[idx]
            n = (idx + 1) // 2
            rt = 0.45 if n <= 3 else 0.09
            show(kind, Ey, Hz, n, rt)
        self.wait(0.6)

        # unstable part
        S2, steps2 = 1.02, 40
        Ey0, Hz0 = self.setup_pulse(S2, 0.7, 24.0)
        bad = yee_run(self.N, S2, steps2, Ey0, Hz0, boundary="pec")
        self.play(e_bars.animate.become(bars(sx, self.E_BASE, bad[0][1], YELLOW)),
                  h_bars.animate.become(bars(sxh, self.H_BASE, bad[0][2] * fd.ETA0, BLUE_B)),
                  status.animate.become(Text(f"Same grid, S = {S2}: unstable", font_size=20, color=RED).to_edge(DOWN, buff=0.75)),
                  step_txt.animate.become(Text("step 0", font_size=20, color=GREY_B).to_edge(DOWN, buff=0.35)), run_time=1.0)
        maxlab = Text("largest |E| = 1.0e+00", font_size=20, color=RED).move_to(_p(4.4, self.E_BASE + 1.15))
        self.add(maxlab)
        for idx in range(2, len(bad), 2):
            _, Ey, Hz = bad[idx]
            n = idx // 2
            big = np.abs(Ey).max()
            self.play(
                e_bars.animate.become(bars(sx, self.E_BASE, Ey, YELLOW)),
                h_bars.animate.become(bars(sxh, self.H_BASE, Hz * fd.ETA0, BLUE_B)),
                step_txt.animate.become(Text(f"step {n}", font_size=20, color=GREY_B).to_edge(DOWN, buff=0.35)),
                maxlab.animate.become(Text(f"largest |E| = {big:.1e}", font_size=20, color=RED).move_to(_p(4.4, self.E_BASE + 1.15))),
                run_time=0.16, rate_func=linear,
            )
        final = Text("the shortest wavelength on the grid grows every step: bars are clipped", font_size=17, color=GREY_B)
        self.play(FadeIn(_fit(final).next_to(status, UP, buff=0.12)), run_time=0.5)
        self.wait(1.5)
