"""Module 3 animation: a capacitor with and without a dielectric.

Left panel: free charge builds on the plates and a uniform field fills the gap.
Right panel: the same plates with a dielectric slab. Molecules align, bound charge appears on the
slab faces, and the field inside drops to E0 / eps_r.

Render with:
  uv run manim -ql --media_dir media/m3_dielectrics animations/m3_dielectrics/scenes.py PolarizationPanels
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY_B,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Create,
    FadeIn,
    Line,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    AnimationGroup,
    LaggedStart,
)

EPS_R = 3.0


def plate(width, y, colour):
    return Line(LEFT * width / 2, RIGHT * width / 2, stroke_width=10, color=colour).shift(UP * y)


def charge_row(width, y, sign, count, colour):
    xs = np.linspace(-width / 2 + 0.25, width / 2 - 0.25, count)
    return VGroup(*[Text(sign, font_size=28, color=colour).move_to([x, y, 0]) for x in xs])


def field_arrows(width, top, bottom, count, length_scale):
    xs = np.linspace(-width / 2 + 0.45, width / 2 - 0.45, count)
    mid = 0.5 * (top + bottom)
    half = 0.5 * (top - bottom) * 0.8 * length_scale
    return VGroup(*[Arrow([x, mid + half, 0], [x, mid - half, 0], buff=0, stroke_width=3, max_tip_length_to_length_ratio=0.2, color=YELLOW) for x in xs])


def molecule(angle, radius=0.17):
    body = Circle(radius=radius, color=GREY_B, stroke_width=2, fill_opacity=0.15)
    plus = Text("+", font_size=18, color=RED).move_to(0.5 * radius * np.array([np.cos(angle), np.sin(angle), 0]))
    minus = Text("-", font_size=20, color=BLUE).move_to(-0.5 * radius * np.array([np.cos(angle), np.sin(angle), 0]))
    return VGroup(body, plus, minus)


class PolarizationPanels(Scene):
    def construct(self):
        width, top, bottom = 5.0, 1.6, -1.6
        left_shift, right_shift = LEFT * 3.4, RIGHT * 3.4
        rng = np.random.default_rng(3)

        # Left panel: vacuum capacitor.
        plates_l = VGroup(plate(width, top, RED), plate(width, bottom, BLUE)).shift(left_shift)
        free_l = VGroup(charge_row(width, top + 0.3, "+", 8, RED), charge_row(width, bottom - 0.3, "-", 8, BLUE)).shift(left_shift)
        field_l = field_arrows(width, top, bottom, 6, 1.0).shift(left_shift)
        title_l = Text("vacuum", font_size=28).next_to(plates_l, UP, buff=0.8)

        # Right panel: dielectric-filled capacitor.
        plates_r = VGroup(plate(width, top, RED), plate(width, bottom, BLUE)).shift(right_shift)
        free_r = VGroup(charge_row(width, top + 0.3, "+", 8, RED), charge_row(width, bottom - 0.3, "-", 8, BLUE)).shift(right_shift)
        slab = Rectangle(width=width - 0.4, height=top - bottom - 0.5, color=ORANGE, stroke_width=2, fill_opacity=0.08).shift(right_shift)
        title_r = Text("dielectric", font_size=28).next_to(plates_r, UP, buff=0.8)

        cols, rows = 7, 4
        xs = np.linspace(-width / 2 + 0.6, width / 2 - 0.6, cols)
        ys = np.linspace(bottom + 0.55, top - 0.55, rows)
        angles = rng.uniform(0, 2 * np.pi, size=(rows, cols))
        molecules = VGroup(*[molecule(angles[i, j]).move_to([x, y, 0]) for i, y in enumerate(ys) for j, x in enumerate(xs)]).shift(right_shift)
        # The field points down, so aligned molecules have the + end down: angle -pi/2.
        aligned = VGroup(*[molecule(-np.pi / 2).move_to([x, y, 0]) for y in ys for x in xs]).shift(right_shift)

        field_r_full = field_arrows(width, top, bottom, 6, 1.0).shift(right_shift)
        field_r_reduced = field_arrows(width, top, bottom, 6, 1.0 / EPS_R).shift(right_shift)
        bound = VGroup(charge_row(width - 0.6, top - 0.42, "-", 7, BLUE), charge_row(width - 0.6, bottom + 0.42, "+", 7, RED)).shift(right_shift)

        caption = Text("same free charge on the plates", font_size=26).to_edge(DOWN)

        self.add(plates_l, plates_r, title_l, title_r, slab, caption)
        self.play(FadeIn(free_l), FadeIn(free_r), run_time=1.2)
        self.play(Create(field_l), Create(field_r_full), run_time=1.2)
        self.wait(0.5)
        self.play(FadeIn(molecules), run_time=1.0)
        self.wait(0.8)

        new_caption = Text("molecules align with the field", font_size=26).to_edge(DOWN)
        self.play(Transform(caption, new_caption), LaggedStart(*[Transform(m, a) for m, a in zip(molecules, aligned)], lag_ratio=0.03), run_time=2.5)
        self.wait(0.6)

        new_caption = Text("bound charge on the faces opposes the free charge", font_size=26).to_edge(DOWN)
        self.play(Transform(caption, new_caption), FadeIn(bound), run_time=1.5)
        self.wait(0.6)

        new_caption = Text(f"inside the dielectric  E = E0 / eps_r  (eps_r = {EPS_R:.0f})", font_size=26).to_edge(DOWN)
        self.play(Transform(caption, new_caption), Transform(field_r_full, field_r_reduced), run_time=2.0)
        self.wait(2.5)
