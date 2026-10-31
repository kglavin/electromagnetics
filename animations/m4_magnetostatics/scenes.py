"""Module 4 animation: field lines of a Helmholtz pair traced by moving particles.

HelmholtzFieldLines  two coaxial loops of radius 1 separated by 1, carrying the same current.
                     Field lines are integrated from B = Biot-Savart sum before the scene starts
                     and drawn as curves; a dot runs along each one with a fading trail while
                     the camera rotates. Lines through the centre leave the box and return
                     outside it, lines near the wires close around them.

Render with:
  uv run manim -ql --media_dir media/m4_magnetostatics animations/m4_magnetostatics/scenes.py HelmholtzFieldLines
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
