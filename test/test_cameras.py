#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Where a camera ends up, given its placement and aim."""
import numpy as np
import pytest

from sionna_rt_gui.cameras import (
    AIM_FREE,
    AIM_LOCKED,
    PLACEMENT_CARRIED,
    PLACEMENT_FREE,
    PLACEMENT_ORBIT,
    ViewCamera,
    resolved_pose,
    safe_view,
)


class FakeGui:
    """Just enough of the app for the pose maths: where things are."""

    def __init__(self, positions=None):
        self.positions = positions or {}

    def look_at_target_position(self, name):
        return self.positions.get(name)

    def scene_center(self):
        return np.zeros(3)


def test_free_camera_keeps_its_pose():
    camera = ViewCamera(
        name="c",
        position=np.array([10.0, 20.0, 30.0]),
        target=np.array([1.0, 2.0, 3.0]),
    )
    position, target = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(position, [10.0, 20.0, 30.0])
    assert np.allclose(target, [1.0, 2.0, 3.0])


def test_orbit_advances_by_speed_and_holds_its_radius():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_ORBIT,
        orbit_center=np.array([5.0, 5.0, 0.0]),
        orbit_radius=100.0,
        orbit_height=40.0,
        orbit_speed_deg_s=90.0,
        orbit_angle_deg=0.0,
    )
    gui = FakeGui()

    # A quarter turn per second at 90 deg/s
    position, target = resolved_pose(gui, camera, 1.0)
    assert camera.orbit_angle_deg == pytest.approx(90.0)
    assert np.allclose(position, [5.0, 105.0, 40.0])
    # It always looks at the centre it is circling
    assert np.allclose(target, [5.0, 5.0, 0.0])

    # And stays that far out, whatever the angle
    for _ in range(7):
        position, _ = resolved_pose(gui, camera, 0.37)
        radius = np.linalg.norm(position[:2] - camera.orbit_center[:2])
        assert radius == pytest.approx(100.0)
        assert position[2] == pytest.approx(40.0)


def test_orbit_angle_wraps():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_ORBIT,
        orbit_speed_deg_s=300.0,
        orbit_angle_deg=200.0,
    )
    resolved_pose(FakeGui(), camera, 1.0)
    assert camera.orbit_angle_deg == pytest.approx(140.0)


def test_carried_camera_keeps_its_offset_and_watches_the_carrier():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_CARRIED,
        carrier="rx-0",
        offset=np.array([0.0, -20.0, 35.0]),
    )
    gui = FakeGui({"rx-0": np.array([100.0, 50.0, 1.5])})
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [100.0, 30.0, 36.5])
    assert np.allclose(target, [100.0, 50.0, 1.5])

    # The offset is kept as the carrier moves
    gui.positions["rx-0"] = np.array([120.0, 50.0, 1.5])
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [120.0, 30.0, 36.5])
    assert np.allclose(target, [120.0, 50.0, 1.5])


def test_carried_camera_falls_back_to_its_own_pose_without_a_carrier():
    camera = ViewCamera(
        name="c",
        position=np.array([7.0, 7.0, 7.0]),
        placement=PLACEMENT_CARRIED,
        carrier="gone",
    )
    position, _ = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(position, [7.0, 7.0, 7.0])


def test_locked_aim_wins_over_the_carrier():
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_CARRIED,
        carrier="rx-0",
        offset=np.array([0.0, 0.0, 50.0]),
        aim=AIM_LOCKED,
        locked_to="tx-0",
    )
    gui = FakeGui(
        {"rx-0": np.array([10.0, 0.0, 1.5]), "tx-0": np.array([-30.0, 40.0, 12.0])}
    )
    position, target = resolved_pose(gui, camera, 0.1)
    # Carried by the receiver, but pointed at the transmitter
    assert np.allclose(position, [10.0, 0.0, 51.5])
    assert np.allclose(target, [-30.0, 40.0, 12.0])


def test_locked_aim_from_a_fixed_position_tracks_a_moving_target():
    camera = ViewCamera(
        name="c",
        position=np.array([0.0, 0.0, 30.0]),
        placement=PLACEMENT_FREE,
        aim=AIM_LOCKED,
        locked_to="rx-0",
    )
    gui = FakeGui({"rx-0": np.array([50.0, 0.0, 1.5])})
    position, target = resolved_pose(gui, camera, 0.1)
    assert np.allclose(position, [0.0, 0.0, 30.0])
    assert np.allclose(target, [50.0, 0.0, 1.5])

    gui.positions["rx-0"] = np.array([50.0, 60.0, 1.5])
    position, target = resolved_pose(gui, camera, 0.1)
    # The camera has not moved, only turned
    assert np.allclose(position, [0.0, 0.0, 30.0])
    assert np.allclose(target, [50.0, 60.0, 1.5])


def test_missing_lock_target_leaves_the_target_alone():
    camera = ViewCamera(
        name="c",
        target=np.array([3.0, 3.0, 3.0]),
        aim=AIM_LOCKED,
        locked_to="not-here",
    )
    _, target = resolved_pose(FakeGui(), camera, 0.1)
    assert np.allclose(target, [3.0, 3.0, 3.0])


def test_straight_down_view_is_tilted_just_enough_to_be_valid():
    # A fly-over asks for an offset that is all height, which looks straight
    # down: collinear with the up axis, which the viewer rejects
    camera = ViewCamera(
        name="c",
        placement=PLACEMENT_CARRIED,
        carrier="rx-0",
        offset=np.array([0.0, 0.0, 120.0]),
    )
    gui = FakeGui({"rx-0": np.array([-19.0, -1.0, 13.0])})
    view = safe_view(*resolved_pose(gui, camera, 0.1))
    assert view is not None
    position, target = view
    delta = target - position
    # Still essentially overhead, but no longer collinear with the up axis
    assert position[2] == pytest.approx(133.0)
    assert np.linalg.norm(delta[:2]) > 1e-3 * abs(delta[2])


def test_a_tilted_view_is_left_alone():
    position, target = safe_view(
        np.array([10.0, -40.0, 60.0]), np.array([10.0, 0.0, 2.0])
    )
    assert np.allclose(position, [10.0, -40.0, 60.0])
    assert np.allclose(target, [10.0, 0.0, 2.0])


def test_a_view_with_nothing_to_look_at_is_refused():
    assert safe_view(np.zeros(3), np.zeros(3)) is None


def test_free_aim_on_a_free_camera_is_not_orbiting():
    camera = ViewCamera(name="c", placement=PLACEMENT_FREE, aim=AIM_FREE)
    before = camera.orbit_angle_deg
    resolved_pose(FakeGui(), camera, 5.0)
    assert camera.orbit_angle_deg == before


class FakeScene:
    def __init__(self, receivers=(), transmitters=(), objects=()):
        self.receivers = {name: None for name in receivers}
        self.transmitters = {name: None for name in transmitters}
        self.objects = {name: None for name in objects}


def test_a_carrier_is_filled_in_so_the_setting_does_something():
    from sionna_rt_gui.cameras import _first_target

    gui = FakeGui()
    gui.scene = FakeScene(receivers=["rx-0"], transmitters=["tx-0"], objects=["car"])
    # A receiver is the most likely thing to follow
    assert _first_target(gui) == "rx-0"

    gui.scene = FakeScene(transmitters=["tx-0"], objects=["car"])
    assert _first_target(gui) == "tx-0"

    gui.scene = FakeScene(objects=["car"])
    assert _first_target(gui) == "car"

    gui.scene = FakeScene()
    assert _first_target(gui) is None


def test_a_view_built_from_nonsense_numbers_is_refused():
    assert safe_view(np.array([np.nan, 0.0, 10.0]), np.zeros(3)) is None
    assert safe_view(np.array([np.inf, 0.0, 10.0]), np.zeros(3)) is None
    assert safe_view(np.zeros(3), np.array([0.0, np.nan, 0.0])) is None


def test_absurd_coordinates_are_held_to_something_a_view_can_use():
    from sionna_rt_gui.cameras import MAX_COORDINATE

    view = safe_view(np.array([1e30, 0.0, 1e30]), np.zeros(3))
    assert view is not None
    position, _ = view
    assert np.all(np.abs(position) <= MAX_COORDINATE)
