"""Shared numerical helpers for the electromagnetics study patch.

Everything here is written from the physics. Nothing is copied from the reference text.
Positions are arrays whose last axis has length 3, so the same functions work for a single
point, a list of points, or a full meshgrid.

show (notebook video embedding and the render command) is imported on use: from emlib import show.
"""

from . import biot_savart, circuits, dielectrics, fdtd, fields, grids, induction, integrators, magnetization, percolation, relax

__all__ = [
    "biot_savart",
    "circuits",
    "dielectrics",
    "fdtd",
    "fields",
    "grids",
    "induction",
    "integrators",
    "magnetization",
    "percolation",
    "relax",
    "show",
]
