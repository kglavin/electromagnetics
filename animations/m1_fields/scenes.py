"""Module 1 animation: a vector field morphing from pure curl to pure divergence.

Render with:  uv run manim -ql animations/m1_fields/scenes.py DivergenceVersusCurl
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
