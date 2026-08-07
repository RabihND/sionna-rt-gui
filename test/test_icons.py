#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Every icon must draw inside the square it is given, otherwise it spills out of
its button. The drawing calls are captured instead of rendered, so this needs no
window.
"""
import numpy as np
import pytest

from sionna_rt_gui import icons

ORIGIN = (40.0, 25.0)
SIZE = 32.0


class RecordingDrawList:
    """Stands in for ImGui's draw list and remembers every point drawn."""

    def __init__(self):
        self.points: list[tuple[float, float]] = []

    def _add(self, *points):
        for point in points:
            self.points.append((float(point[0]), float(point[1])))

    def AddLine(self, a, b, color, thickness=1.0):
        self._add(a, b)

    def AddRect(self, a, b, color, rounding=0.0, flags=0, thickness=1.0):
        self._add(a, b)

    def AddRectFilled(self, a, b, color, rounding=0.0, flags=0):
        self._add(a, b)

    def AddCircle(self, center, radius, color, segments=0, thickness=1.0):
        self._add(
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
        )

    def AddCircleFilled(self, center, radius, color, segments=0):
        self.AddCircle(center, radius, color)

    def AddPolyline(self, points, color, flags, thickness):
        array = np.asarray(points).reshape(-1, 2)
        self._add(*array)

    def AddText(self, position, color, text):
        self._add(position)


ICON_FUNCTIONS = [
    icons.transmitter,
    icons.receiver,
    icons.car,
    icons.truck,
    icons.person,
    icons.drone,
    icons.tree,
    icons.wall,
    icons.glass,
    icons.radio_map,
    icons.paths,
    icons.antenna,
    icons.antenna_lobe,
    icons.object_icon,
    icons.scene_icon,
    icons.render_icon,
    icons.timeline_icon,
    icons.impulse_icon,
]


@pytest.mark.parametrize("draw", ICON_FUNCTIONS, ids=lambda f: f.__name__)
def test_icon_stays_inside_its_square(draw):
    recorder = RecordingDrawList()
    draw(recorder, ORIGIN[0], ORIGIN[1], SIZE, 0xFFFFFFFF, 1.0)

    assert recorder.points, "the icon drew nothing"
    xs = np.array([p[0] for p in recorder.points])
    ys = np.array([p[1] for p in recorder.points])
    # A hair of tolerance for stroke width, but no more
    tolerance = 0.02 * SIZE
    assert xs.min() >= ORIGIN[0] - tolerance, f"spills left by {ORIGIN[0] - xs.min():.1f}px"
    assert ys.min() >= ORIGIN[1] - tolerance, f"spills above by {ORIGIN[1] - ys.min():.1f}px"
    assert xs.max() <= ORIGIN[0] + SIZE + tolerance, (
        f"spills right by {xs.max() - ORIGIN[0] - SIZE:.1f}px"
    )
    assert ys.max() <= ORIGIN[1] + SIZE + tolerance, (
        f"spills below by {ys.max() - ORIGIN[1] - SIZE:.1f}px"
    )


def test_every_asset_has_an_icon():
    from sionna_rt_gui.assets import ASSET_LIBRARY

    for spec in ASSET_LIBRARY:
        assert spec.key in icons.ASSET_ICONS, f"{spec.key} has no icon"
