"""Module 2 animations.

RelaxationSurface   a 2D potential relaxing from a flat guess to the Laplace solution
GaussianSphereFlux  a point charge inside a sphere, surface arrows coloured by E . dS

Render with:
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py RelaxationSurface
  uv run manim -ql --media_dir media/m2_electrostatics animations/m2_electrostatics/scenes.py GaussianSphereFlux
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
