"""Module 3 animations.

PolarizationPanels: a capacitor with and without a dielectric. Left panel: free charge builds on the
plates and a uniform field fills the gap. Right panel: the same plates with a dielectric slab.
Molecules align, bound charge appears on the slab faces, and the field inside drops to E0 / eps_r.

DisplacedBalls (primer 00a, section 3.3): a uniformly polarized sphere as two uniformly charged balls
of density +rho and -rho displaced by d. The overlap is neutral, the uncovered crescents are the
surface charge P cos(theta), and the two interior ball fields add to the uniform -P / 3 eps0.

Render with:
  uv run manim -ql --media_dir media/m3_dielectrics animations/m3_dielectrics/scenes.py PolarizationPanels
  uv run manim -ql --media_dir media/m3_dielectrics animations/m3_dielectrics/scenes.py DisplacedBalls
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


# DisplacedBalls ---------------------------------------------------------------------------------
# Scene units with eps0 = 1. Inside a uniformly charged ball of density rho centred at c the field is
# rho (r - c) / 3. The +rho ball sits at O + d/2 (up, along P) and the -rho ball at O - d/2.

from manim import (  # noqa: E402
    GREY_D,
    Axes,
    Dot,
    FadeOut,
    Intersection,
    ValueTracker,
    always_redraw,
)

BALL_R = 2.0
BALL_O = np.array([-3.4, -0.45, 0.0])
P_FIXED = 0.5                       # P = rho d, held fixed in the last act
D_START = 0.5                       # displacement for the first acts, so rho = 1
FIELD_SCALE = 3.6                   # scene length per unit field
SAMPLE_OFFSETS = np.array([[-1.0, 0.9], [0.0, 0.9], [1.0, 0.9], [-1.3, 0.0], [0.0, 0.0], [1.3, 0.0],
                           [-1.0, -0.9], [0.0, -0.9], [1.0, -0.9]])


def ball_field(rho, centre, pts):
    """Interior field rho (r - c) / 3 of a uniformly charged ball (eps0 = 1), pts (N, 2)."""
    return rho * (pts - centre[None, :2]) / 3.0


def interior_field(rho, d, pts):
    up = BALL_O[:2] + np.array([0.0, d / 2])
    down = BALL_O[:2] - np.array([0.0, d / 2])
    return ball_field(rho, up, pts) + ball_field(-rho, down, pts)


class DisplacedBalls(Scene):
    def construct(self):
        d = ValueTracker(0.0)
        fixed_p = ValueTracker(0.0)          # 0: rho = 1 while the balls separate; 1: rho = P / d in the last act
        pts = BALL_O[:2] + SAMPLE_OFFSETS

        def rho_val():
            return P_FIXED / max(d.get_value(), 1e-6) if fixed_p.get_value() > 0.5 else 1.0

        title = Text("A uniformly polarized sphere = two displaced balls of charge", font_size=30).to_edge(UP)

        def centre(sign):
            return BALL_O + sign * np.array([0.0, d.get_value() / 2, 0.0])

        def crescent_opacity():
            return min(0.9, 0.45 * rho_val() ** 0.3)

        pos_ball = always_redraw(lambda: Circle(BALL_R, color=RED, stroke_width=2, fill_color=RED,
                                                fill_opacity=crescent_opacity()).move_to(centre(+1)))
        neg_ball = always_redraw(lambda: Circle(BALL_R, color=BLUE, stroke_width=2, fill_color=BLUE,
                                                fill_opacity=crescent_opacity()).move_to(centre(-1)))
        overlap = always_redraw(lambda: Intersection(Circle(BALL_R).move_to(centre(+1)), Circle(BALL_R).move_to(centre(-1)),
                                                     color=GREY_D, fill_color=GREY_D, fill_opacity=1.0, stroke_width=0))
        centres = always_redraw(lambda: VGroup(Dot(centre(+1), radius=0.05, color=RED), Dot(centre(-1), radius=0.05, color=BLUE)))
        p_arrow = VGroup(Arrow(BALL_O + [-2.75, -0.6, 0], BALL_O + [-2.75, 0.6, 0], buff=0, color=WHITE, stroke_width=4),
                         Text("P", font_size=28).move_to(BALL_O + [-2.75, 0.85, 0]))

        def readout_rho_d():
            r, dd = rho_val(), d.get_value()
            return Text(f"ρ = {r:.2f}    d = {dd:.2f}    P = ρd = {r * dd:.2f}", font_size=24).move_to([3.6, 2.55, 0])

        caption = Text("+ρ and −ρ balls on top of each other: neutral everywhere, no field", font_size=24).to_edge(DOWN)
        self.add(title)
        self.play(FadeIn(pos_ball), FadeIn(neg_ball), FadeIn(overlap), FadeIn(centres), FadeIn(caption), run_time=1.2)
        self.wait(1.2)

        # Act 1: slide the balls apart.
        rd = always_redraw(readout_rho_d)
        new = Text("slide +ρ up and −ρ down by d: the overlap (grey) stays neutral", font_size=24).to_edge(DOWN)
        self.play(FadeIn(rd), FadeIn(p_arrow), Transform(caption, new), run_time=1.0)
        self.play(d.animate.set_value(D_START), run_time=3.0)
        self.wait(0.6)

        # Act 2: the crescents are the surface charge.
        top = BALL_O + [0, BALL_R + D_START / 2 + 0.28, 0]
        bot = BALL_O - [0, BALL_R + D_START / 2 + 0.28, 0]
        signs = VGroup(Text("+", font_size=34, color=RED).move_to(top), Text("−", font_size=34, color=BLUE).move_to(bot))
        th = np.radians(50)
        radial = Line(BALL_O, BALL_O + BALL_R * np.array([np.sin(th), np.cos(th), 0]), color=WHITE, stroke_width=2)
        th_label = Text("θ", font_size=24).move_to(BALL_O + 0.7 * np.array([np.sin(th / 2), np.cos(th / 2), 0]))
        crescent_text = VGroup(
            Text("uncovered crescent: thickness d cos θ", font_size=24),
            Text("so σ_b = ρ d cos θ = P cos θ", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.6, 1.75, 0])

        axes = Axes(x_range=[0, 180, 30], y_range=[-1, 1, 0.5], x_length=4.6, y_length=1.9,
                    axis_config={"include_numbers": False, "color": GREY_B, "stroke_width": 2}, tips=False).move_to([3.6, 0.1, 0])
        curve = axes.plot(lambda t: np.cos(np.radians(t)), color=ORANGE, x_range=[0, 180])
        ax_labels = VGroup(Text("σ_b / P", font_size=20).next_to(axes, LEFT, buff=0.15),
                           Text("θ from 0° (top) to 180° (bottom)", font_size=18).next_to(axes, DOWN, buff=0.12),
                           Text("+1", font_size=16).next_to(axes.c2p(0, 1), LEFT, buff=0.08),
                           Text("−1", font_size=16).next_to(axes.c2p(0, -1), LEFT, buff=0.08))
        new = Text("the uncovered crescents are the bound surface charge", font_size=24).to_edge(DOWN)
        self.play(FadeIn(signs), Create(radial), FadeIn(th_label), Transform(caption, new), run_time=1.2)
        self.play(FadeIn(crescent_text), Create(axes), FadeIn(ax_labels), run_time=1.2)
        self.play(Create(curve), run_time=1.5)
        self.wait(1.5)

        # Act 3: the interior field from the two balls.
        self.play(FadeOut(crescent_text), FadeOut(axes), FadeOut(ax_labels), FadeOut(curve), FadeOut(radial), FadeOut(th_label), run_time=0.8)
        rule = VGroup(Text("inside each ball: E = ρ r_c / 3ε0, from its own centre", font_size=22),
                      Text("red: from the +ρ ball   blue: from the −ρ ball", font_size=22),
                      Text("yellow: their sum", font_size=22, color=YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, 1.6, 0])
        self.play(FadeIn(rule), run_time=0.8)

        r0, d0 = 1.0, D_START
        cp, cm = BALL_O[:2] + [0, d0 / 2], BALL_O[:2] - [0, d0 / 2]
        demo = [0, 5, 7]
        reds, blues, yellows = VGroup(), VGroup(), VGroup()
        for i in demo:
            p = pts[i]
            e_plus = ball_field(r0, cp, p[None])[0] * FIELD_SCALE
            e_minus = ball_field(-r0, cm, p[None])[0] * FIELD_SCALE
            p3 = np.array([p[0], p[1], 0.0])
            reds.add(Arrow(p3, p3 + [*e_plus, 0], buff=0, color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.15))
            blues.add(Arrow(p3 + [*e_plus, 0], p3 + [*(e_plus + e_minus), 0], buff=0, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.15))
            yellows.add(Arrow(p3, p3 + [*(e_plus + e_minus), 0], buff=0, color=YELLOW, stroke_width=6, max_tip_length_to_length_ratio=0.3))
        new = Text("each ball alone gives a radial field that grows with distance from its centre", font_size=24).to_edge(DOWN)
        self.play(Create(reds), Transform(caption, new), run_time=1.5)
        self.wait(0.5)
        self.play(Create(blues), run_time=1.5)
        self.wait(0.5)
        new = Text("the r_c terms cancel: E_in = −ρd / 3ε0 = −P / 3ε0 at every point", font_size=24).to_edge(DOWN)
        self.play(Create(yellows), Transform(caption, new), run_time=1.2)
        self.wait(1.2)

        def all_arrows():
            E = interior_field(rho_val(), d.get_value(), pts) * FIELD_SCALE
            return VGroup(*[Arrow([p[0], p[1], 0], [p[0] + e[0], p[1] + e[1], 0], buff=0, color=YELLOW,
                                  stroke_width=6, max_tip_length_to_length_ratio=0.3) for p, e in zip(pts, E)])

        def e_readout():
            r, dd = rho_val(), d.get_value()
            E = interior_field(r, dd, pts)
            ratio = np.linalg.norm(E, axis=1) / (r * dd)
            down = np.all(E[:, 1] < 0) and np.allclose(E[:, 0], 0.0, atol=1e-12)
            return VGroup(Text(f"|E_in| / (P/ε0) at all 9 points: {ratio.mean():.3f}", font_size=24),
                          Text(f"spread {ratio.max() - ratio.min():.1e},  direction {'−P (down)' if down else 'mixed'}", font_size=22)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.5, 0.3, 0])

        field = always_redraw(all_arrows)
        er = always_redraw(e_readout)
        self.play(FadeOut(reds), FadeOut(blues), FadeOut(yellows), FadeIn(field), FadeIn(er), run_time=1.2)
        self.wait(1.5)

        # Act 4: the surface-layer limit at fixed P.
        new = Text("shrink d and raise ρ with P = ρd fixed: a thin surface layer, the same interior field", font_size=22).to_edge(DOWN)
        self.play(FadeOut(rule), Transform(caption, new), run_time=0.8)
        fixed_p.set_value(1.0)               # at d = D_START this still gives rho = 1, so nothing jumps
        signs.add_updater(lambda m: m[0].move_to(BALL_O + [0, BALL_R + d.get_value() / 2 + 0.28, 0]))
        signs.add_updater(lambda m: m[1].move_to(BALL_O - [0, BALL_R + d.get_value() / 2 + 0.28, 0]))
        self.play(d.animate.set_value(0.06), run_time=4.0)
        self.wait(1.0)

        closing = VGroup(Text("inside: E = −P / 3ε0, uniform", font_size=24),
                         Text("outside: a dipole field, p = (4/3)πR³P", font_size=24)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.5, -1.4, 0])
        self.play(FadeIn(closing), run_time=1.0)
        self.wait(2.5)
