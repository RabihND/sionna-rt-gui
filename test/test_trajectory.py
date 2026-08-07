#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Waypoint editing must keep the cumulative distances consistent."""
import numpy as np
import pytest

from sionna_rt_gui.animation import Trajectory


def straight_line():
    trajectory = Trajectory()
    for point in ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 20.0, 0.0]):
        trajectory.add_point(point)
    return trajectory


def test_total_distance_follows_the_waypoints():
    assert straight_line().total_distance() == pytest.approx(30.0)


def test_position_interpolates_along_the_path():
    trajectory = straight_line()
    trajectory.distance = 5.0
    position, direction = trajectory.current_position_and_direction()
    assert position == pytest.approx([5.0, 0.0, 0.0])
    assert direction == pytest.approx([1.0, 0.0, 0.0])

    trajectory.distance = 20.0
    position, direction = trajectory.current_position_and_direction()
    assert position == pytest.approx([10.0, 10.0, 0.0])
    assert direction == pytest.approx([0.0, 1.0, 0.0])


def test_removing_a_waypoint_recomputes_the_distances():
    trajectory = straight_line()
    trajectory.remove_point(1)
    assert len(trajectory) == 2
    # Straight from the origin to the last point
    assert trajectory.total_distance() == pytest.approx(np.hypot(10.0, 20.0))


def test_editing_a_waypoint_recomputes_the_distances():
    trajectory = straight_line()
    trajectory.set_point(1, [20.0, 0.0, 0.0])
    # Legs are now 20 m out and then back in to the last point
    expected = 20.0 + float(np.linalg.norm([10.0, 20.0]))
    assert trajectory.total_distance() == pytest.approx(expected)


def test_the_distance_stays_within_the_path_after_editing():
    trajectory = straight_line()
    trajectory.distance = 30.0
    trajectory.remove_point(2)
    assert trajectory.distance <= trajectory.total_distance()


def test_an_empty_trajectory_has_no_position():
    assert Trajectory().current_position_and_direction() is None
