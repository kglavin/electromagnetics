"""Shared numerical helpers for the electromagnetics study patch.

Everything here is written from the physics. Nothing is copied from the reference text.
Positions are arrays whose last axis has length 3, so the same functions work for a single
point, a list of points, or a full meshgrid.
"""

from . import biot_savart, circuits, dielectrics, fields, grids, integrators, magnetization, percolation, relax, show

__all__ = [
    "biot_savart",
    "circuits",
    "dielectrics",
    "fields",
    "grids",
    "integrators",
    "magnetization",
    "percolation",
    "relax",
    "show",
]
