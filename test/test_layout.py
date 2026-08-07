#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""The docked areas must tile the window without gaps or overlaps."""
import pytest

from sionna_rt_gui.workspace_layout import AreaLayout

WINDOW = (1920, 1080)


def areas(scale=1.0, **overrides):
    layout = AreaLayout(**overrides)
    return layout, layout.areas(WINDOW, scale)


def test_areas_cover_the_window_without_overlapping():
    _, rects = areas()
    width, height = WINDOW

    topbar, status = rects["topbar"], rects["status"]
    assert topbar[1] == 0.0
    assert status[1] + status[3] == pytest.approx(height)
    assert topbar[2] == pytest.approx(width)

    # The middle band is shared by the tools, the centre and the side column
    tools, viewport, side = rects["tools"], rects["viewport"], rects["outliner"]
    assert tools[0] == 0.0
    assert viewport[0] == pytest.approx(tools[0] + tools[2])
    assert side[0] == pytest.approx(viewport[0] + viewport[2])
    assert side[0] + side[2] == pytest.approx(width)

    # The outliner and properties split the side column exactly
    properties = rects["properties"]
    assert properties[1] == pytest.approx(side[1] + side[3])
    assert properties[1] + properties[3] == pytest.approx(tools[1] + tools[3])

    # The timeline sits under the viewport, together filling the middle band
    timeline = rects["timeline"]
    assert timeline[1] == pytest.approx(viewport[1] + viewport[3])
    assert timeline[0] == pytest.approx(viewport[0])
    assert timeline[2] == pytest.approx(viewport[2])


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_areas_scale_with_the_interface(scale):
    layout, rects = areas(scale)
    assert rects["topbar"][3] == pytest.approx(layout.topbar_height * scale)
    assert rects["tools"][2] == pytest.approx(layout.tools_width * scale)


def test_a_dragged_border_cannot_swallow_the_window():
    # What an extreme splitter drag would set
    layout, rects = areas(tools_width=5000.0, side_width=5000.0, timeline_height=5000.0)
    assert 0.0 <= layout.tools_width <= 0.3 * WINDOW[0]
    assert layout.side_width <= 0.6 * WINDOW[0]
    assert layout.timeline_height <= 0.5 * WINDOW[1]
    # The viewport keeps a positive size
    assert rects["viewport"][2] > 0.0
    assert rects["viewport"][3] > 0.0


def test_borders_cannot_be_dragged_to_nothing():
    layout, _ = areas(side_width=-100.0, outliner_fraction=-1.0)
    assert layout.side_width >= 180.0
    assert 0.1 <= layout.outliner_fraction <= 0.85
