"""Module 2 animations.

RelaxationSurface   a 2D potential relaxing from a flat guess to the Laplace solution
GaussianSphereFlux  a point charge inside a sphere, surface arrows coloured by E . dS
JacobiModes         five-point Jacobi and Gauss-Seidel by hand, then sine modes decaying under Jacobi
ConductorEquilibrium mobile charge in a slab moves until the field inside vanishes
ImageCharge         a charge approaching a grounded plane, its image and the induced surface charge
SolidAngleFlux      a cone from the charge cuts a sphere and a lumpy surface: equal flux per solid angle; outside, pairs cancel

Render with:
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py RelaxationSurface
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py GaussianSphereFlux
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py JacobiModes
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py ConductorEquilibrium
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py ImageCharge
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py SolidAngleFlux
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from manim import (
    BLUE,
    BLUE_E,
    DEGREES,
    DOWN,
    GREY,
    ORIGIN,
    OUT,
    RED,
    UL,
    UP,
    WHITE,
    YELLOW,
    Arrow3D,
    Dot3D,
    Scene,
    Sphere,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    interpolate_color,
    linear,
    LEFT,
    RIGHT,
    DR,
    UR,
    GREY_B,
    ORANGE,
    Arrow,
    Axes,
    Circle,
    DashedLine,
    DashedVMobject,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    VMobject,
    Write,
    Dot,
    Polygon,
)

from emlib import relax


def relaxation_snapshots(n=31, every=1, max_sweeps=400):
    """Return a list of potential grids after 0, every, 2*every, ... Gauss-Seidel sweeps.

    Problem: grounded square box, top edge at 1 V, a grounded square block inside.
    """
    V = np.zeros((n, n))
    fixed = np.zeros((n, n), dtype=bool)
    V[:, -1] = 1.0
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    a, b = int(0.35 * n), int(0.55 * n)
    fixed[a:b, a:b] = True
    src = np.zeros((n, n))
    snaps = [V.copy()]
    for k in range(1, max_sweeps + 1):
        change = relax.gauss_seidel_sweep(V, fixed, src, 1.0)
        if k % every == 0:
            snaps.append(V.copy())
        if change < 1e-5:
            break
    return snaps


class RelaxationSurface(ThreeDScene):
    def construct(self):
        snaps = relaxation_snapshots(n=31, every=2)
        n = snaps[0].shape[0]
        k = ValueTracker(0.0)

        axes = ThreeDAxes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], z_range=[0, 1, 0.5],
                          x_length=5, y_length=5, z_length=2.5)

        grid = np.linspace(0, 1, n)

        def surface_at(frame):
            interp = RegularGridInterpolator((grid, grid), snaps[frame], method="linear")

            def point(u, v):
                return axes.c2p(u, v, float(interp([[u, v]])[0]))

            return Surface(point, u_range=[0, 1], v_range=[0, 1], resolution=(n - 1, n - 1),
                           fill_opacity=0.9, stroke_width=0.3, stroke_color=WHITE).set_fill_by_value(
                axes=axes, colorscale=[(BLUE_E, 0.0), (YELLOW, 0.5), (RED, 1.0)], axis=2)

        surface = always_redraw(lambda: surface_at(int(round(k.get_value()))))
        def label_text():
            return Text(f"Gauss-Seidel sweep {2 * int(round(k.get_value()))}", font_size=30).to_corner(UL)

        label = label_text()
        self.add_fixed_in_frame_mobjects(label)

        def update_label(m):
            # become() copies glyphs into m's own submobjects and may add new ones. Only mobjects in the
            # camera's fixed-in-frame set stay 2D, so register m's current family after every rebuild.
            m.become(label_text())
            self.add_fixed_in_frame_mobjects(m)

        label.add_updater(update_label)

        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES, zoom=0.9)
        self.add(axes, surface)
        self.wait(1)
        self.play(k.animate.set_value(len(snaps) - 1), run_time=8, rate_func=linear)
        self.wait(1)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(4)


class GaussianSphereFlux(ThreeDScene):
    """Arrows are built at three keyframes (charge centred, off-centre, outside) and transformed
    between them. Rebuilding 3D arrows every frame is far too slow, so this uses 32 low-resolution
    arrows and lets Transform interpolate their positions and colours."""

    def construct(self):
        R = 2.0
        m = 32
        idx = np.arange(m) + 0.5
        phi = np.arccos(1 - 2 * idx / m)
        theta = np.pi * (1 + 5**0.5) * idx
        normals = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
        points = R * normals

        def arrows(charge_x):
            c = np.array([charge_x, 0.0, 0.0])
            r = points - c
            E = r / np.linalg.norm(r, axis=1, keepdims=True) ** 3
            En = np.einsum("ij,ij->i", E, normals)
            scale = 1.0 / (np.abs(En).max() + 1e-12)
            group = VGroup()
            for p, nvec, e in zip(points, normals, En):
                length = 0.9 * abs(e) * scale + 0.15
                colour = interpolate_color(BLUE, RED, 0.5 + 0.5 * float(np.clip(e * scale, -1, 1)))
                group.add(Arrow3D(p, p + np.sign(e) * length * nvec, color=colour, thickness=0.015,
                                  height=0.18, base_radius=0.06, resolution=6))
            return group

        sphere = Sphere(radius=R, resolution=(16, 32), fill_opacity=0.12, stroke_width=0.3, color=GREY)
        charge = Dot3D([0.0, 0.0, 0.0], radius=0.12, color=YELLOW)
        field = arrows(0.0)

        inside = Text("charge inside: every patch has E . dS > 0, flux = q / eps0", font_size=26).to_edge(DOWN)
        outside = Text("charge outside: flux in on one side, out on the other, total 0", font_size=26).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(inside)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.9)
        self.add(sphere, field, charge)
        self.wait(1.5)
        self.play(Transform(field, arrows(1.2)), charge.animate.move_to([1.2, 0, 0]), run_time=3)
        self.wait(1.5)
        self.remove(inside)
        self.add_fixed_in_frame_mobjects(outside)
        self.play(Transform(field, arrows(3.2)), charge.animate.move_to([3.2, 0, 0]), run_time=3)
        self.wait(2)


# ---------------------------------------------------------------------------
# Primer 00b, section 3.4: relaxation by averaging, and why long modes are slow.


def jacobi_1d(V):
    """One Jacobi sweep of the 1D Laplace average with fixed end values (the primer's update)."""
    Vn = V.copy()
    Vn[1:-1] = 0.5 * (V[:-2] + V[2:])
    return Vn


def gauss_seidel_1d(V):
    V = V.copy()
    for i in range(1, len(V) - 1):
        V[i] = 0.5 * (V[i - 1] + V[i + 1])
    return V


def five_point_histories(tol=1e-6):
    """Jacobi and Gauss-Seidel histories on the five-point grid of primer 00b, worked examples 4 and 5."""
    out = {}
    for name, sweep in (("Jacobi", jacobi_1d), ("Gauss-Seidel", gauss_seidel_1d)):
        V = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        hist, changes = [V.copy()], [np.nan]
        while True:
            Vn = sweep(V)
            changes.append(np.abs(Vn - V).max())
            hist.append(Vn.copy())
            V = Vn
            if changes[-1] < tol:
                break
        out[name] = (hist, changes)
    return out


def mode_history(N=50, ks=(1, 5, 20), amps=(1.0, 0.6, 0.4), shown=(0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)):
    """Error curves and sine-mode amplitudes under Jacobi on N interior points with zero ends."""
    h = 1.0 / (N + 1)
    x = np.arange(N + 2) * h
    e = sum(a * np.sin(k * np.pi * x) for k, a in zip(ks, amps))
    curves, amplitudes = {}, {}
    for n in range(max(shown) + 1):
        if n in shown:
            curves[n] = e.copy()
            # amplitude of mode k by the discrete sine transform: (2/(N+1)) sum e_i sin(k pi x_i)
            amplitudes[n] = [2 * h * np.sum(e * np.sin(k * np.pi * x)) for k in ks]
        e = jacobi_1d(e)
    factors = [np.cos(k * np.pi * h) for k in ks]
    return x, curves, amplitudes, factors, h


class JacobiModes(Scene):
    """Shapes are transformed between keyframes; text is swapped instantly so numbers never morph."""

    def bars(self, V, origin, scale, colour):
        shapes, labels = VGroup(), VGroup()
        for i, v in enumerate(V):
            fixed = i in (0, len(V) - 1)
            x = origin[0] + 1.1 * i
            height = max(v * scale, 0.02)
            bar = Rectangle(width=0.7, height=height, stroke_width=1, fill_opacity=0.8,
                            color=GREY if fixed else colour, fill_color=GREY if fixed else colour)
            bar.move_to([x, origin[1] + height / 2, 0])
            shapes.add(bar)
            labels.add(Text(f"{v:.3f}", font_size=20).next_to(bar, UP, buff=0.08))
        return shapes, labels

    def swap(self, old, new):
        self.remove(old)
        self.add(new)
        return new

    def construct(self):
        title = Text("Relaxation: every point becomes the average of its neighbours", font_size=28).to_edge(UP)
        self.add(title)

        # Part one: the five-point grid, Jacobi on top, Gauss-Seidel below.
        hist = five_point_histories()
        scale = 1.6
        rows = {"Jacobi": np.array([-4.6, 0.6, 0]), "Gauss-Seidel": np.array([-4.6, -2.9, 0])}
        colours = {"Jacobi": BLUE, "Gauss-Seidel": ORANGE}
        shapes, labels, info = {}, {}, {}
        for name, origin in rows.items():
            shapes[name], labels[name] = self.bars(hist[name][0][0], origin, scale, colours[name])
            heading = Text(name, font_size=24).move_to(origin + np.array([-1.4, 1.9, 0]), aligned_edge=LEFT)
            base = Line(origin + np.array([-0.5, 0, 0]), origin + np.array([4.9, 0, 0]), stroke_width=1, color=GREY_B)
            self.add(heading, base, shapes[name], labels[name])
        self.add(Text("exact: 0, 0.25, 0.5, 0.75, 1", font_size=22, color=GREY_B).move_to([3.6, 2.5, 0]))

        def info_text(name, k):
            ch = hist[name][1]
            change = "" if k == 0 else f"largest change {ch[k]:.4f}"
            return Text(f"sweep {k}   {change}", font_size=22).move_to(rows[name] + np.array([8.2, 1.0, 0]))

        for name in rows:
            info[name] = info_text(name, 0)
            self.add(info[name])
        self.wait(1.5)
        n_final = {name: len(hist[name][0]) - 1 for name in rows}
        for k in list(range(1, 9)) + ["final"]:
            new_shapes, new_labels, new_info = {}, {}, {}
            for name, origin in rows.items():
                kk = n_final[name] if k == "final" else k
                new_shapes[name], new_labels[name] = self.bars(hist[name][0][kk], origin, scale, colours[name])
                new_info[name] = (Text(f"converged after {kk} sweeps (change < 1e-6)", font_size=22).move_to(origin + np.array([8.2, 1.0, 0]))
                                  if k == "final" else info_text(name, kk))
                self.remove(labels[name])
            self.play(*[Transform(shapes[n], new_shapes[n]) for n in rows], run_time=0.8 if k in (1, 2, 3, "final") else 0.4)
            for name in rows:
                labels[name] = self.swap(labels[name], new_labels[name])
                info[name] = self.swap(info[name], new_info[name])
            self.wait(1.2 if k in (1, 2, 3) else 0.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])

        # Part two: the error is a sum of sine modes, and Jacobi damps mode k by cos(k pi h) per sweep.
        title2 = Text("The error on 50 points: three sine modes under Jacobi sweeps", font_size=28).to_edge(UP)
        self.play(FadeOut(title), FadeIn(title2))
        x, curves, amplitudes, factors, h = mode_history()
        ax = Axes(x_range=[0, 1, 0.25], y_range=[-1.6, 1.6, 0.5], x_length=7.0, y_length=4.2,
                  axis_config={"include_tip": False, "stroke_width": 2}).move_to([-2.6, -0.3, 0])
        xl = Text("x", font_size=22).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = Text("error", font_size=22).next_to(ax.y_axis, UP, buff=0.1)
        self.add(ax, xl, yl)

        def curve(n):
            pts = [ax.c2p(xi, yi) for xi, yi in zip(x, curves[n])]
            return VMobject(color=YELLOW, stroke_width=3).set_points_as_corners(pts)

        ks = (1, 5, 20)
        bar_base = np.array([2.6, -1.5, 0])
        bar_colours = [BLUE, ORANGE, RED]

        def mode_bars(n):
            g, t = VGroup(), VGroup()
            for j, (k, a, a0) in enumerate(zip(ks, amplitudes[n], amplitudes[0])):
                frac = abs(a) / abs(a0)
                bx = bar_base + np.array([1.4 * j, 0, 0])
                height = max(3.0 * frac, 0.02)
                g.add(Rectangle(width=0.8, height=height, fill_opacity=0.8, color=bar_colours[j], fill_color=bar_colours[j],
                                stroke_width=1).move_to(bx + np.array([0, height / 2, 0])))
                t.add(Text(f"k = {k}", font_size=20).move_to(bx + np.array([0, -0.3, 0])),
                      Text(f"{frac:.1e}" if frac > 1e-12 else "< 1e-12", font_size=18).move_to(bx + np.array([0, -0.65, 0])))
            return g, t

        def counter(n):
            return Text(f"sweep {n}", font_size=24).move_to([-2.6, 2.4, 0])

        heading = Text("amplitude / start", font_size=22).move_to(bar_base + np.array([1.4, 3.6, 0]))
        factor_txt = Text("damping per sweep cos(kπh): " + ", ".join(f"{f:.4f}" for f in factors), font_size=20).to_edge(DOWN).shift(UP * 0.35)
        cur = curve(0)
        bars_g, bars_t = mode_bars(0)
        count = counter(0)
        self.play(FadeIn(cur), FadeIn(bars_g), FadeIn(bars_t), FadeIn(heading), FadeIn(count), FadeIn(factor_txt))
        self.wait(1.5)
        for n in sorted(curves)[1:]:
            new_g, new_t = mode_bars(n)
            self.remove(bars_t)
            self.play(Transform(cur, curve(n)), Transform(bars_g, new_g), run_time=0.8)
            bars_t = self.swap(bars_t, new_t)
            count = self.swap(count, counter(n))
            self.wait(0.5)
        n_long = int(np.ceil(np.log(1e-6) / np.log(factors[0])))
        caption = Text(f"short modes vanish in a few sweeps; k = 1 needs {n_long} sweeps to fall to 1e-6, a count that grows like N²",
                       font_size=20, color=GREY_B).to_edge(DOWN)
        self.play(FadeOut(factor_txt), FadeIn(caption))
        self.wait(3)


# ---------------------------------------------------------------------------
# Primer 00c, section 3.1: a conductor in an applied field.

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12


class ConductorEquilibrium(Scene):
    def construct(self):
        title = Text("A conductor in a field: mobile charge moves until E inside is zero", font_size=28).to_edge(UP)
        self.add(title)
        W, H, ncol, nrow = 4.2, 4.0, 6, 6
        d = W / ncol
        slab = Rectangle(width=W, height=H, color=GREY, fill_color=GREY, fill_opacity=0.15, stroke_width=3).shift(DOWN * 0.3)
        self.add(slab)
        cx, cy = slab.get_center()[0], slab.get_center()[1]
        xs = cx - W / 2 + (np.arange(ncol) + 0.5) * d
        ys = cy - H / 2 + (np.arange(nrow) + 0.5) * (H / nrow)
        ions = VGroup(*[Text("+", font_size=26, color=RED).move_to([x, y, 0]) for x in xs for y in ys])
        start, target = [], []
        for j, x in enumerate(xs):
            for i, y in enumerate(ys):
                start.append(np.array([x + 0.2, y, 0]))
                if j == 0:  # the leftmost electrons pile up against the left face
                    target.append(np.array([cx - W / 2 + 0.1, y, 0]))
                else:  # every other electron moves one column left and pairs with the neighbouring ion
                    target.append(np.array([xs[j - 1] + 0.2, y, 0]))
        start, target = np.array(start), np.array(target)
        s = ValueTracker(0.0)
        electrons = VGroup(*[Circle(radius=0.07, color=BLUE, fill_color=BLUE, fill_opacity=1, stroke_width=0).move_to(p) for p in start])
        for k, e in enumerate(electrons):
            e.add_updater(lambda m, k=k: m.move_to(start[k] + s.get_value() * (target[k] - start[k])))
        self.add(ions, electrons)
        neutral = Text("red +: fixed ions    blue dots: mobile electrons    neutral: every ion has an electron beside it", font_size=20).to_edge(DOWN)
        self.add(neutral)
        self.wait(1.5)

        # Applied field E0 along +x, outside on both sides; inside it is E0 minus the field of the separated charge.
        ys_arrow = cy + np.linspace(-1.5, 1.5, 4)

        def outside_arrows():
            g = VGroup()
            for y in ys_arrow:
                for x0 in (-6.4, -4.2, cx + W / 2 + 0.4, cx + W / 2 + 2.6):
                    g.add(Arrow([x0, y, 0], [x0 + 1.5, y, 0], buff=0, color=YELLOW, stroke_width=4))
            return g

        def inside_arrows():
            frac = 1.0 - s.get_value()  # E_in / E0 = 1 - sigma / (eps0 E0), and sigma grows linearly with the shift
            g = VGroup()
            for y in ys_arrow:
                for x0 in (cx - 1.1, cx + 0.4):
                    if frac > 0.02:
                        g.add(Arrow([x0, y + 0.3, 0], [x0 + 0.9 * frac, y + 0.3, 0], buff=0, color=YELLOW, stroke_width=4,
                                    max_tip_length_to_length_ratio=0.35))
            return g

        e_label = Text("applied field E0", font_size=22, color=YELLOW).move_to([-5.2, cy + 2.3, 0])
        self.play(FadeIn(outside_arrows()), FadeIn(e_label), FadeOut(neutral))
        inside = always_redraw(inside_arrows)
        self.play(FadeIn(inside))
        readout = always_redraw(lambda: Text(f"E inside / E0 = {1 - s.get_value():.2f}", font_size=26)
                                .move_to([cx + W / 2 + 1.9, cy + 2.3, 0]))
        self.add(readout)
        self.wait(0.5)
        self.play(s.animate.set_value(1.0), run_time=6, rate_func=linear)
        for e in electrons:
            e.clear_updaters()
        self.remove(inside)
        minus = Text("net −", font_size=24, color=BLUE).next_to(slab, DOWN, buff=0.15).align_to(slab, LEFT)
        plus = Text("net +", font_size=24, color=RED).next_to(slab, DOWN, buff=0.15).align_to(slab, RIGHT)
        right_ions = VGroup(*[ions[(ncol - 1) * nrow + i] for i in range(nrow)])
        self.play(FadeIn(minus), FadeIn(plus), *[ion.animate.scale(1.5) for ion in right_ions],
                  *[electrons[i].animate.scale(1.5) for i in range(nrow)])
        # how far the electrons really move: sigma = eps0 E0 from a copper electron density
        E0, n_cu = 1e5, 8.5e28
        shift = EPS0 * E0 / (n_cu * E_CHARGE)
        cap = Text(f"interior stays neutral; only the faces are charged. Shift exaggerated: for E0 = 100 kV/m in copper it is {shift:.1e} m",
                   font_size=18, color=GREY_B).to_edge(DOWN)
        self.play(FadeIn(cap))
        self.wait(3)


# ---------------------------------------------------------------------------
# Primer 00c, section 3.3: the method of images for a grounded plane.

K_COULOMB = 1.0 / (4 * np.pi * EPS0)


class ImageCharge(Scene):
    def construct(self):
        title = Text("A charge above a grounded plane, its image, and the induced charge", font_size=28).to_edge(UP)
        self.add(title)
        q_nC = 1.0
        px, py, unit = -3.6, -0.2, 0.75  # plane centre on screen, and screen units per cm
        plane = Line([px - 3.0, py, 0], [px + 3.0, py, 0], stroke_width=8, color=GREY)
        ground = Text("grounded plane, V = 0", font_size=20, color=GREY_B).next_to(plane, DOWN, buff=0.15).align_to(plane, LEFT)
        d = ValueTracker(3.0)
        self.add(plane)

        def real():
            return VGroup(Circle(radius=0.22, color=RED, fill_opacity=0.9).move_to([px, py + unit * d.get_value(), 0]),
                          Text("+q", font_size=22).move_to([px + 0.55, py + unit * d.get_value(), 0]))

        def image():
            c = DashedVMobject(Circle(radius=0.22, color=BLUE), num_dashes=12).move_to([px, py - unit * d.get_value(), 0])
            return VGroup(c, Text("−q image", font_size=20, color=BLUE).move_to([px + 0.95, py - unit * d.get_value(), 0]))

        def link():
            return DashedLine([px, py + unit * d.get_value() - 0.25, 0], [px, py - unit * d.get_value() + 0.25, 0],
                              color=GREY_B, stroke_width=2)

        def sigma(s_cm, dd):  # induced charge per area in units of q per cm^2
            return -dd / (2 * np.pi * (s_cm**2 + dd**2) ** 1.5)

        def plane_arrows():  # field at the plane points into it, with strength proportional to |sigma|
            g = VGroup()
            dd = d.get_value()
            peak = abs(sigma(0, 0.5))
            for s_cm in np.linspace(-3.15, 3.15, 10):  # no arrow at s = 0, where the charge sits
                L = 1.2 * abs(sigma(s_cm, dd)) / peak
                if L > 0.05:
                    x = px + unit * s_cm
                    g.add(Arrow([x, py + L + 0.05, 0], [x, py + 0.05, 0], buff=0, color=YELLOW, stroke_width=3,
                                max_tip_length_to_length_ratio=0.3))
            return g

        ax = Axes(x_range=[-4, 4, 2], y_range=[-0.7, 0, 0.35], x_length=5.0, y_length=3.0,
                  axis_config={"include_tip": False, "stroke_width": 2, "include_numbers": False}).move_to([3.6, 0.6, 0])
        ax_lab = Text("σ(s) / q   [1/cm²]", font_size=20).next_to(ax, UP, buff=0.15)
        s_lab = Text("s [cm] along the plane", font_size=20).next_to(ax, DOWN, buff=0.15)
        ticks = VGroup(*[Text(f"{v:g}", font_size=16).next_to(ax.c2p(-4, v), LEFT, buff=0.1) for v in (0, -0.35, -0.7)])

        def profile():
            ss = np.linspace(-4, 4, 161)
            return VMobject(color=BLUE, stroke_width=4).set_points_as_corners([ax.c2p(si, sigma(si, d.get_value())) for si in ss])

        def readout():
            dd = d.get_value()
            F = K_COULOMB * (q_nC * 1e-9) ** 2 / (2 * dd * 1e-2) ** 2
            return VGroup(Text(f"d = {dd:.2f} cm", font_size=22),
                          Text(f"σ(0) = {sigma(0, dd):.3f} q/cm²", font_size=22),
                          Text(f"pull on q = {F:.2e} N  (q = 1 nC)", font_size=22)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.6, -2.15, 0])

        objs = [always_redraw(f) for f in (real, image, link, plane_arrows, profile, readout)]
        self.add(ground, ax, ax_lab, s_lab, ticks, *objs)
        note = Text("total induced charge = −q at every d; V = 0 on the plane because the two distances are equal",
                    font_size=18, color=GREY_B).to_edge(DOWN)
        self.add(note)
        self.wait(1.5)
        self.play(d.animate.set_value(1.0), run_time=5, rate_func=linear)
        self.wait(1.5)
        self.play(d.animate.set_value(0.5), run_time=3, rate_func=linear)
        self.wait(3)


# ---------------------------------------------------------------------------------------------
# SolidAngleFlux: why the flux of a point charge does not depend on the surface.
#
# All lengths are in screen units and all fluxes in units of q / (4 pi eps0), so the flux through
# a patch equals the solid angle it subtends. The scene shows a slice through the charge. The
# cone is DPHI wide in the page and DELTA deep out of it, so its solid angle is DPHI * DELTA.
# The patch it cuts on a surface is measured from the drawn geometry: the in-page length comes
# from clipping the polygon to the wedge, the out-of-page width is r * DELTA, and the flux is
# E . n dS = (1 / r^2) cos(alpha) dS.

SA_DPHI = 0.1
SA_DELTA = 0.1


def _closed_polygon(P):
    """Vertices Q (closed), segment midpoints, lengths and outward normals of a counter-clockwise polygon."""
    Q = np.vstack([P, P[:1]])
    d = np.diff(Q, axis=0)
    L = np.linalg.norm(d, axis=1)
    normal = np.stack([d[:, 1], -d[:, 0]], axis=-1) / L[:, None]
    return Q, 0.5 * (Q[1:] + Q[:-1]), L, normal


def _lumpy(centre, n=3000):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 2.0 + 0.5 * np.cos(3 * t + 0.3) + 0.25 * np.sin(5 * t + 1.0)
    return centre + np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)


def _wedge(poly, src, phi_c, dphi):
    """Per-segment weight (fraction of the segment's angular extent inside the wedge) and, per segment,
    distance r, cos(alpha) with sign (r_hat . n), and full flux (r_hat . n) L / r * DELTA."""
    Q, mid, L, normal = poly
    rel = Q - src
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    b0 = (ang[:-1] - phi_c + np.pi) % (2 * np.pi) - np.pi
    b1 = b0 + (ang[1:] - ang[:-1] + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.minimum(b0, b1), np.maximum(b0, b1)
    overlap = np.clip(np.minimum(hi, dphi / 2) - np.maximum(lo, -dphi / 2), 0.0, None)
    w = np.where(hi - lo > 1e-15, overlap / np.maximum(hi - lo, 1e-15), 0.0)
    rm = mid - src
    r = np.linalg.norm(rm, axis=1)
    rn = np.einsum("ij,ij->i", rm / r[:, None], normal)
    return w, r, rn, rn * L / r * SA_DELTA


def _patches(poly, src, phi_c, dphi=SA_DPHI):
    """The patches the cone cuts on a polygon, nearest first. Each: dict with flux, r, alpha, length, points."""
    Q, _, L, _ = poly
    w, r, rn, c = _wedge(poly, src, phi_c, dphi)
    idx = np.nonzero(w > 0)[0]
    if idx.size == 0:
        return []
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(w) - 1:
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    out = []
    for run in runs:
        ww = w[run]
        length = np.sum(ww * L[run])
        if length < 1e-9:
            continue
        flux = np.sum(ww * c[run])
        rbar = np.sum(ww * L[run] * r[run]) / length
        cos_a = abs(np.sum(ww * rn[run] * L[run])) / length
        pts = np.vstack([Q[run], Q[run[-1] + 1]])
        out.append(dict(flux=flux, r=rbar, cos=cos_a, length=length, points=pts))
    out.sort(key=lambda p: p["r"])
    return out


def _cumulative(poly, src, n=1440):
    """Accumulated flux over wedges from phi = 0 to each grid angle, split into positive and negative parts."""
    edges = np.linspace(0.0, 2 * np.pi, n + 1)
    pos, neg = np.zeros(n), np.zeros(n)
    for j in range(n):
        w, _, _, c = _wedge(poly, src, 0.5 * (edges[j] + edges[j + 1]), edges[1] - edges[0])
        f = w * c
        pos[j], neg[j] = f[f > 0].sum(), f[f < 0].sum()
    return edges, np.concatenate([[0.0], np.cumsum(pos)]), np.concatenate([[0.0], np.cumsum(neg)])


def _p3(xy):
    return np.array([xy[0], xy[1], 0.0])


class SolidAngleFlux(Scene):
    def construct(self):
        C = np.array([-3.3, -0.4])
        R = 1.1
        t = np.linspace(0.0, 2 * np.pi, 3000, endpoint=False)
        lump = _closed_polygon(_lumpy(C))
        circ = _closed_polygon(C + R * np.stack([np.cos(t), np.sin(t)], axis=-1))
        outside = C + np.array([-3.2, 0.35])
        sx, sy = ValueTracker(C[0]), ValueTracker(C[1])
        phi = ValueTracker(0.0)
        show_cone = ValueTracker(1.0)
        cone_len = ValueTracker(3.4)

        def src():
            return np.array([sx.get_value(), sy.get_value()])

        title = Text("Why the flux is the same through any closed surface around the charge", font_size=26).to_edge(UP)
        lump_mob = VMobject(color=GREY_B, stroke_width=3).set_points_as_corners([_p3(p) for p in lump[0][::10]] + [_p3(lump[0][0])])
        circ_mob = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(_p3(C))
        lump_lab = Text("lumpy surface", font_size=20, color=GREY_B).move_to(_p3(C + np.array([1.55, 2.3])))
        circ_lab = Text("sphere", font_size=20, color=GREY_B).move_to(_p3(C + np.array([0.0, -0.75])))
        note = Text("A slice through the charge. The cone is 0.1 rad wide in the page and 0.1 rad deep out of it: ΔΩ = 0.0100 sr.",
                    font_size=18, color=GREY_B).to_edge(DOWN)

        def field_lines():
            s = src()
            return VGroup(*[Line(_p3(s), _p3(s + 9.0 * np.array([np.cos(a), np.sin(a)])), stroke_width=1, color=GREY, stroke_opacity=0.35)
                            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False) + 0.07])

        def charge():
            return Dot(_p3(src()), radius=0.11, color=RED)

        def cone():
            if show_cone.get_value() < 0.5:
                return VGroup()
            s, a = src(), phi.get_value()
            L = cone_len.get_value()
            p1 = s + L * np.array([np.cos(a - SA_DPHI / 2), np.sin(a - SA_DPHI / 2)])
            p2 = s + L * np.array([np.cos(a + SA_DPHI / 2), np.sin(a + SA_DPHI / 2)])
            fill = Polygon(_p3(s), _p3(p1), _p3(p2), stroke_width=0, fill_color=YELLOW, fill_opacity=0.18)
            return VGroup(fill, Line(_p3(s), _p3(p1), color=YELLOW, stroke_width=2), Line(_p3(s), _p3(p2), color=YELLOW, stroke_width=2))

        surfaces = {"circ": circ, "lump": lump}
        active = {"circ": True, "lump": True}

        def highlights():
            g = VGroup()
            if show_cone.get_value() < 0.5:
                return g
            for key, poly in surfaces.items():
                if not active[key]:
                    continue
                for p in _patches(poly, src(), phi.get_value()):
                    g.add(VMobject(color=RED if p["flux"] > 0 else BLUE, stroke_width=9).set_points_as_corners([_p3(q) for q in p["points"]]))
            return g

        x0 = 0.55
        fs = 19

        def z(v):  # print a rounded zero without a minus sign
            return round(v, 4) + 0.0
        cum_c = _cumulative(circ, C)
        cum_l = _cumulative(lump, C)

        def acc(cum, a):
            edges, pos, neg = cum
            return np.interp(a, edges, pos), np.interp(a, edges, neg)

        def readout_inside():
            a = phi.get_value()
            pc = _patches(circ, C, a)
            pl = _patches(lump, C, a)
            lines = [Text("flux through the patch  [q/(4πε0)]:", font_size=fs)]
            if pc and pl:
                c, l = pc[0], pl[0]
                lines += [
                    Text(f"sphere:  r = {c['r']:.2f}   α = {np.degrees(np.arccos(min(c['cos'], 1))):2.0f}°   E·n ΔS = {c['flux']:.4f}", font_size=fs, color=RED),
                    Text(f"lumpy:   r = {l['r']:.2f}   α = {np.degrees(np.arccos(min(l['cos'], 1))):2.0f}°   E·n ΔS = {l['flux']:.4f}", font_size=fs, color=RED),
                    Text("lumpy patch compared with the sphere patch:", font_size=fs),
                    Text(f"area ×{l['length'] * l['r'] / (c['length'] * c['r']):.2f}    E ×{(c['r'] / l['r']) ** 2:.3f}    cos α = {l['cos']:.2f}",
                         font_size=fs),
                    Text(f"product ×{l['flux'] / c['flux']:.3f}", font_size=fs),
                ]
            fc, _ = acc(cum_c, a)
            fl, _ = acc(cum_l, a)
            lines += [Text("accumulated over the band swept so far:", font_size=fs),
                      Text(f"sphere {fc:.3f}    lumpy {fl:.3f}    swept Ω = {a * SA_DELTA:.3f} sr", font_size=fs, color=YELLOW)]
            return VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([x0, 0.2, 0], aligned_edge=LEFT)

        # Part one: charge inside both surfaces.
        bg = always_redraw(field_lines)
        self.add(title, bg, circ_mob, lump_mob, circ_lab, lump_lab, note)
        cone_m, hl_m, q_m = always_redraw(cone), always_redraw(highlights), always_redraw(charge)
        self.add(cone_m, hl_m, q_m)
        ro = always_redraw(readout_inside)
        self.add(ro)
        self.wait(1.5)
        self.play(phi.animate.set_value(2 * np.pi), run_time=11, rate_func=linear)
        self.remove(ro)
        final = readout_inside()
        self.add(final)
        concl = Text("This band is 0.628 of the 4π sr around the charge.\n"
                     "Every band gives equal fluxes, so over all directions\n"
                     "both surfaces carry 4π · q/(4πε0) = q/ε0.", font_size=fs, color=YELLOW
                     ).next_to(final, DOWN, buff=0.35).align_to(final, LEFT)
        self.play(FadeIn(concl), run_time=0.8)
        self.wait(3.5)

        # Part two: move the charge outside the lumpy surface.
        show_cone.set_value(0.0)
        active["circ"] = False
        title2 = Text("Charge outside: every line of sight crosses the surface an even number of times", font_size=26).to_edge(UP)
        self.play(FadeOut(final), FadeOut(concl), FadeOut(circ_mob), FadeOut(circ_lab), Transform(title, title2), run_time=1.0)
        self.play(sx.animate.set_value(outside[0]), sy.animate.set_value(outside[1]), run_time=2.0)
        cum_o = _cumulative(lump, outside)

        def readout_outside():
            a = phi.get_value()
            ps = _patches(lump, outside, a)
            lines = [Text("patches cut by the cone, nearest first  [q/(4πε0)]:", font_size=fs)]
            if not ps:
                lines.append(Text("this cone misses the surface", font_size=fs, color=GREY_B))
            for k, p in enumerate(ps[:4]):
                kind = "enters" if p["flux"] < 0 else "leaves"
                lines.append(Text(f"crossing {k + 1}: {p['flux']:+.4f}   {kind}", font_size=fs, color=BLUE if p["flux"] < 0 else RED))
            lines.append(Text(f"sum along this cone: {z(sum(p['flux'] for p in ps)):+.4f}", font_size=fs))
            pos, neg = acc(cum_o, a)
            lines += [Text("accumulated over the band swept so far:", font_size=fs),
                      Text(f"entering {neg:+.3f}    leaving {pos:+.3f}    net {z(pos + neg):+.3f}", font_size=fs, color=YELLOW)]
            return VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([x0, 0.2, 0], aligned_edge=LEFT)

        phi.set_value(0.0)
        cone_len.set_value(6.4)
        show_cone.set_value(1.0)
        ro2 = always_redraw(readout_outside)
        self.add(ro2)
        self.wait(1.0)
        self.play(phi.animate.set_value(2 * np.pi), run_time=11, rate_func=linear)
        self.remove(ro2)
        final2 = readout_outside()
        self.add(final2)
        concl2 = Text("Each entering patch is cancelled by a leaving\n"
                      "patch on the same line of sight. A charge outside\n"
                      "contributes nothing to the flux.", font_size=fs, color=YELLOW
                      ).next_to(final2, DOWN, buff=0.35).align_to(final2, LEFT)
        self.play(FadeIn(concl2), run_time=0.8)
        self.wait(3.5)
