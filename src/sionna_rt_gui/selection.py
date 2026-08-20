#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from enum import Enum

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.utils.geometry import rotation_matrix

from .animation import trajectory_gui
from .config import DEFAULT_SLICE_PLANE_NAME
from .geo_ref import enu_to_gps
from .pattern_viz import default_pattern_scale, update_antenna_pattern_structure
from .ps_utils import ACCENT_BRIGHT
from .workspace_layout import (
    end_property_row,
    numeric_field,
    numeric_field3,
    property_row,
)
from .sionna_utils import set_or_update_radio_devices_polyscope


class SelectionType(Enum):
    Transmitter = "Transmitter"
    Receiver = "Receiver"
    RadioMap = "Radio map"
    Path = "Path"
    Mesh = "Mesh"


GIZMO_SCALE = 40


def gps_property_row(gui, position) -> None:
    """Show the position as GPS coordinates when the scene is geo-referenced
    (a BBOX.json sidecar next to the scene file, see geo_ref.py)."""
    origin = getattr(gui, "scene_geo", None)
    if origin is None:
        return
    lat, lon, alt = enu_to_gps(
        origin, float(position[0]), float(position[1]), float(position[2])
    )
    property_row("GPS", gui.ui_scale)
    psim.Text(f"{lat:.6f}, {lon:.6f} | {alt:.1f} m ASL")
    end_property_row()


def vec_str(vec: np.ndarray) -> str:
    vec = vec.squeeze()
    return f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"


def selection_gui(
    gui: "SionnaRtGui",
    selected_object: rt.SceneObject | None,
    selected_type: SelectionType | None,
):
    """Selection properties in their own floating window (classic layout)."""
    if selected_object is None:
        return

    # Place window in the top-right corner of the screen
    window_resolution = ps.get_window_size()
    w, h = 430, 540
    psim.SetNextWindowSize(
        (w * gui.ui_scale, h * gui.ui_scale), psim.ImGuiCond_FirstUseEver
    )
    # Top right corner
    window_pos = (
        window_resolution[0] - (w + 10) * gui.ui_scale,
        10 * gui.ui_scale,
    )
    psim.SetNextWindowPos(window_pos, psim.ImGuiCond_FirstUseEver)

    # Dynamic title: show what is selected (the "###" suffix keeps the
    # window identity stable while the visible title changes).
    object_name = getattr(selected_object, "name", "")
    _, keep_selection = psim.Begin(
        f"{selected_type.value}: {object_name}###sionna_selection", open=True
    )

    if not keep_selection:
        psim.End()
        gui.clear_selection()
        return

    selection_contents(gui, selected_object, selected_type)
    psim.End()


def material_contents(gui: "SionnaRtGui", scene_object: rt.SceneObject) -> None:
    """
    Radio material of an object: pick any material already in the scene, and
    edit the electromagnetic properties of the custom ones.
    """
    material = scene_object.radio_material
    names = sorted(gui.scene.radio_materials.keys())
    current_name = getattr(material, "name", "")
    current_index = names.index(current_name) if current_name in names else 0

    property_row("Material", gui.ui_scale)
    changed, new_index = psim.Combo("##material", current_index, names)
    end_property_row()
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Materials are shared by name across the scene, so editing one\n"
            "affects every object that uses it."
        )
    if changed and names[new_index] != current_name:
        scene_object.radio_material = gui.scene.radio_materials[names[new_index]]
        gui.on_scene_materials_changed()
        return

    if isinstance(material, rt.ITURadioMaterial):
        psim.TextDisabled(f"ITU-R P.2040 type: {material.itu_type}")
    else:
        # A custom material: its properties are ours to edit
        material_changed = False
        property_row("Permittivity", gui.ui_scale)
        changed, value = psim.DragFloat(
            "##permittivity",
            float(material.relative_permittivity[0]),
            0.1,
            1.0,
            100.0,
            format="%.2f",
        )
        end_property_row()
        if changed:
            material.relative_permittivity = value
            material_changed = True

        property_row("Conductivity [S/m]", gui.ui_scale)
        changed, value = psim.DragFloat(
            "##conductivity",
            float(material.conductivity[0]),
            0.01,
            0.0,
            100.0,
            format="%.3f",
        )
        end_property_row()
        if changed:
            material.conductivity = value
            material_changed = True

        if material_changed:
            gui.on_scene_materials_changed()

    property_row("Scattering", gui.ui_scale)
    changed, value = psim.SliderFloat(
        "##scattering",
        float(material.scattering_coefficient[0]),
        0.0,
        1.0,
        format="%.2f",
    )
    end_property_row()
    if changed:
        material.scattering_coefficient = value
        gui.on_scene_materials_changed()


def scene_object_contents(gui: "SionnaRtGui", scene_object: rt.SceneObject) -> None:
    """
    Properties of a plain scene object (a building, a placed asset, ...):
    editable position and orientation, its radio material, and a move gizmo.
    """
    name = scene_object.name
    psim.Text(name)
    psim.SameLine()
    if name in gui.placed_assets:
        bw = 80 * gui.ui_scale
        psim.SetCursorPosX(
            psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - bw - 5
        )
        pressed_del = not psim.IsAnyItemActive() and psim.IsKeyPressed(
            psim.ImGuiKey_Delete, repeat=False
        )
        if psim.Button("Remove##object", (bw, 0)) or pressed_del:
            gui.remove_asset(name)
            gui.clear_selection()
            return
    else:
        psim.TextDisabled("(scene geometry)")

    extents = np.array(scene_object.mi_mesh.bbox().extents())
    psim.TextDisabled(
        f"Size: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} m"
    )
    psim.Spacing()
    material_contents(gui, scene_object)

    psim.Spacing()
    position = scene_object.position.numpy().squeeze()
    property_row("Position [m]", gui.ui_scale)
    changed, new_position = numeric_field3(
        "##object_position", position, 0.25, "%.2f"
    )
    end_property_row()
    gps_property_row(gui, position)
    edited_numerically = False
    if changed:
        gui.set_object_position(scene_object, new_position)
        edited_numerically = True

    orientation_deg = np.degrees(scene_object.orientation.numpy().squeeze())
    property_row("Orientation [deg]", gui.ui_scale)
    changed, new_orientation = numeric_field3(
        "##object_orientation", orientation_deg, 1.0, "%.1f"
    )
    end_property_row()
    if changed:
        gui.set_object_orientation(scene_object, np.radians(new_orientation).tolist())
        edited_numerically = True

    psim.Spacing()
    psim.TextDisabled("Drag the gizmo in the 3D view to move this object.")

    # --- Move gizmo. The gizmo widget follows the structure's transform, so the
    # transform (not the point) is what places it on the object. Drags are
    # applied as deltas, so no absolute reference pose has to be tracked.
    is_new_gizmo = not ps.has_point_cloud("Gizmo")
    if is_new_gizmo:
        struct = ps.register_point_cloud("Gizmo", np.zeros((1, 3)), enabled=False)
        gizmo = struct.get_transformation_gizmo()
        gizmo.set_enabled(True)
        gizmo.set_allow_scaling(False)
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
    else:
        struct = ps.get_point_cloud("Gizmo")

    to_world = struct.get_transform()
    previous = gui.object_gizmo_previous
    if previous is not None and not np.allclose(previous, to_world):
        # The gizmo was dragged: move the object by the same amount
        translation = to_world[:3, 3] - previous[:3, 3]
        rotation = to_world[:3, :3] @ np.linalg.inv(previous[:3, :3])
        gui.transform_scene_object(
            scene_object, translation=translation, rotation_increment=rotation
        )
        gui.object_gizmo_previous = to_world
        return

    # The gizmo is idle, so keep it on the object. This also catches moves that
    # did not come from the gizmo: typed coordinates, trajectories, attachments.
    target = np.eye(4)
    target[:3, :3] = rotation_matrix(scene_object.orientation).numpy()[..., 0]
    target[:3, 3] = scene_object.position.numpy().squeeze()
    if previous is None or not np.allclose(target, to_world):
        struct.set_transform(target)
    gui.object_gizmo_previous = target


def selection_contents(
    gui: "SionnaRtGui",
    selected_object: rt.SceneObject | None,
    selected_type: SelectionType | None,
):
    """
    Properties of the selected object, drawn into the current window so they
    can be shown either floating or inside a docked area.
    """
    if selected_object is None:
        psim.PushTextWrapPos(0.0)
        psim.TextDisabled(
            "No active object. Click a radio device or a scene object "
            "in the 3D view to edit it here."
        )
        psim.PopTextWrapPos()
        return

    if selected_type == SelectionType.Mesh:
        scene_object_contents(gui, selected_object)
        return

    rd_update_needed = False
    is_transmitter = selected_type == SelectionType.Transmitter
    if selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
        rd = selected_object
        array = gui.scene.tx_array if is_transmitter else gui.scene.rx_array
        pattern = array.antenna_pattern

        changed, rd.color = psim.ColorEdit3(
            "Color##selection",
            rd.color,
            psim.ImGuiColorEditFlags_NoInputs,
        )
        rd_update_needed |= changed

        psim.SameLine()
        # "Remove" button (right-aligned)
        bw = 80
        psim.SetCursorPosX(
            psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - bw - 5
        )
        pressed_del = not psim.IsAnyItemActive() and psim.IsKeyPressed(
            psim.ImGuiKey_Delete, repeat=False
        )
        if psim.Button("Remove", (bw, 0)) or pressed_del:
            gui.remove_object(selected_object, selected_type)
            rd_update_needed = True
            gui.clear_selection()

        psim.NewLine()

        # TODO: do not trigger constant GPU -> CPU transfers
        if psim.TreeNodeEx(
            "Characteristics:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            # Reserve room for the widget labels, which would otherwise be
            # clipped when the window is narrow.
            position = rd.position.numpy().squeeze()
            property_row("Position [m]", gui.ui_scale)
            changed, new_position = numeric_field3(
                "##position", position, 0.25, "%.2f"
            )
            end_property_row()
            gps_property_row(gui, position)
            if changed:
                rd.position = mi.Point3f(*new_position)
                dr.make_opaque(rd.position)
                rd_update_needed = True

            orientation_deg = np.degrees(rd.orientation.numpy().squeeze())
            is_tracking = rd.name in gui.look_at_targets
            psim.BeginDisabled(is_tracking)
            property_row("Orientation [deg]", gui.ui_scale)
            changed, new_orientation = numeric_field3(
                "##orientation", orientation_deg, 1.0, "%.1f"
            )
            end_property_row()
            psim.EndDisabled()
            if is_tracking:
                changed = False
            if changed:
                # Note: mi.Point3f rejects numpy scalar types, convert to float
                rd.orientation = mi.Point3f(*np.radians(new_orientation).tolist())
                dr.make_opaque(rd.orientation)
                rd_update_needed = True

            if is_transmitter:
                power_dbm = float(rd.power_dbm[0])
                watts = 10.0 ** (power_dbm / 10.0) / 1000.0
                property_row("TX power [dBm]", gui.ui_scale)
                changed, new_power_dbm = numeric_field(
                    "##tx_power",
                    power_dbm,
                    0.25,
                    -30.0,
                    60.0,
                    "%.1f",
                    f"{watts:.3g} W",
                )
                end_property_row()
                if changed:
                    rd.power_dbm = new_power_dbm
                    rd_update_needed = True

            velocity = rd.velocity.numpy().squeeze()
            property_row("Velocity [m/s]", gui.ui_scale)
            changed, new_velocity = numeric_field3(
                "##velocity",
                velocity,
                0.1,
                "%.2f",
                "World-space velocity, used for Doppler",
            )
            end_property_row()
            if changed:
                rd.velocity = mi.Vector3f(*[float(v) for v in new_velocity])
                dr.make_opaque(rd.velocity)
                rd_update_needed = True
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "World-space velocity vector, used for Doppler.\n"
                    "Independent of the orientation. While a trajectory\n"
                    "plays, it is overwritten with the travel velocity."
                )
            psim.TreePop()

        # --- Keep the device pointed at something
        target_names = ["(none)"]
        target_names += [
            name
            for name in list(gui.scene._transmitters.keys())
            + list(gui.scene._receivers.keys())
            if name != rd.name
        ]
        target_names += list(gui.scene.objects.keys())
        current_target = gui.look_at_targets.get(rd.name)
        target_index = (
            target_names.index(current_target)
            if current_target in target_names
            else 0
        )
        property_row("Look at", gui.ui_scale)
        changed, new_target = psim.Combo("##look_at", target_index, target_names)
        end_property_row()
        if psim.IsItemHovered():
            psim.SetTooltip(
                "Keep this device pointed at another device or an object.\n"
                "It follows the target as either of them moves, so the\n"
                "orientation is no longer yours to set by hand."
            )
        if changed:
            gui.set_look_at_target(
                rd, None if new_target == 0 else target_names[new_target]
            )
            rd_update_needed = True
        if current_target is not None:
            psim.TextDisabled(f"Tracking {current_target}")

        # --- Attach the device to a scene object (e.g. a vehicle)
        object_names = ["(none)"] + list(gui.scene.objects.keys())
        attachment = gui.attachments.get(rd.name)
        current_index = 0
        if attachment is not None and attachment["object"] in object_names:
            current_index = object_names.index(attachment["object"])
        property_row("Attach to object", gui.ui_scale)
        changed, new_index = psim.Combo(
            "##attach_object", current_index, object_names
        )
        end_property_row()
        if psim.IsItemHovered():
            psim.SetTooltip(
                "Snap the device on top of a scene object. The object then\n"
                "follows the device: give the device a trajectory and the\n"
                "object drives along with it (physics included)."
            )
        if changed:
            gui.attach_device_to_object(
                rd, None if new_index == 0 else object_names[new_index]
            )
            rd_update_needed = True

        if gui.attach_pick_pending == rd.name:
            hovered = gui.attach_hover_name
            psim.TextColored(
                (*ACCENT_BRIGHT, 1.0),
                f"Click to attach to '{hovered}' (Esc to cancel)"
                if hovered
                else "Hover an object in the 3D scene (Esc to cancel)",
            )
        else:
            if psim.Button("Pick object in scene##attach"):
                gui.start_attach_pick(rd.name)
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "Then click the object (e.g. a car) directly in the\n"
                    "3D view. The device snaps just above the clicked point."
                )

        psim.Spacing()
        if psim.TreeNodeEx(
            "Antenna array:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Text(
                f"Type: {type(array).__name__}\n"
                f"Array size: {dr.width(array.normalized_positions)}\n"
                f"Pattern: {type(pattern).__name__}\n"
            )

            changed, gui.show_antenna_pattern = psim.Checkbox(
                "Show 3D radiation pattern##antenna",
                getattr(gui, "show_antenna_pattern", False),
            )
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "Radiation pattern of a single antenna element, drawn\n"
                    "at the device: radius is the linear gain, color is\n"
                    "the gain in dB. Follows the device's orientation."
                )
            if gui.show_antenna_pattern:
                psim.PushItemWidth(-140 * gui.ui_scale)
                if gui.antenna_pattern_scale <= 0.0:
                    gui.antenna_pattern_scale = default_pattern_scale(gui)
                _, gui.antenna_pattern_scale = psim.SliderFloat(
                    "Pattern size [m]##antenna",
                    gui.antenna_pattern_scale,
                    1.0,
                    max(4.0 * default_pattern_scale(gui), 60.0),
                    format="%.1f",
                    flags=psim.ImGuiSliderFlags_Logarithmic,
                )
                psim.PopItemWidth()
                _, gui.antenna_pattern_db_radius = psim.Checkbox(
                    "Radius in dB##antenna",
                    getattr(gui, "antenna_pattern_db_radius", False),
                )
                if psim.IsItemHovered():
                    psim.SetTooltip(
                        "Radius proportional to gain in dB (40 dB range),\n"
                        "like MATLAB's pattern(): weak side lobes become\n"
                        "visible. Off: radius is the linear gain, as in\n"
                        "sionna.rt's AntennaPattern.show()."
                    )

            _, gui.show_pattern_cuts = psim.Checkbox(
                "Show 2D pattern cuts##antenna",
                getattr(gui, "show_pattern_cuts", False),
            )
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "Polar charts of the vertical and horizontal gain\n"
                    "cuts, like sionna.rt's AntennaPattern.show()."
                )

            psim.TreePop()

        psim.Spacing()
        if psim.TreeNodeEx(
            "Animation:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            rd_update_needed |= trajectory_gui(gui, selected_object)
            psim.TreePop()

        # --- Transformation gizmo
        # Check that the selection wasn't cleared before updating the gizmo.
        if gui.selected_object is not None:
            # TODO: make RD's orientation visible while the gizmo is shown.
            if not ps.has_point_cloud("Gizmo"):
                struct = ps.register_point_cloud(
                    "Gizmo", rd.position.numpy().T, enabled=False
                )
                gizmo = struct.get_transformation_gizmo()
                gizmo.set_enabled(True)
                gizmo.set_allow_scaling(False)
                struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)
            else:
                struct = ps.get_point_cloud("Gizmo")

            to_world = struct.get_transform()

            # Gizmo moved since the last frame
            if (gui.prev_gizmo_to_world is not None) and not np.allclose(
                gui.prev_gizmo_to_world, to_world
            ):
                # Apply transform to the selected object
                rd.position = mi.Point3f(to_world[:3, -1])
                target = to_world[:3, -1] + (
                    to_world[:3, 0] / np.linalg.norm(to_world[:3, 0])
                )
                rd.look_at(target)
                # For debugging:
                # ps.register_point_cloud("Gizmo target", target[None, :], color=(1, 0, 1))
                dr.make_opaque(rd.position, rd.orientation)
                rd_update_needed = True

            else:
                # Reset position & orientation of gizmo to match the selected object
                to_world = np.eye(4)
                to_world[:3, :3] = rotation_matrix(rd.orientation).numpy()[..., 0]
                to_world[:3, -1] = rd.position.numpy().squeeze()
                struct.set_transform(to_world)

            gui.prev_gizmo_to_world = to_world

            # Antenna pattern preview, following the device's pose
            update_antenna_pattern_structure(
                gui, rd, array, getattr(gui, "show_antenna_pattern", False)
            )

    psim.Spacing()

    if rd_update_needed:
        # TODO: auto-pause animation if animated object was moved?

        if gui.selected_object is not None:
            # Carry along any scene object attached to this device
            gui.update_attached_object(gui.selected_object)

        set_or_update_radio_devices_polyscope(
            gui.scene.transmitters if is_transmitter else gui.scene.receivers,
            is_transmitter=is_transmitter,
            gui=gui,
        )
        if is_transmitter:
            # Note: receivers don't affect radio maps.
            gui.reset_radio_map()

        if gui.cfg.paths.auto_update:
            gui.update_paths(clear_first=True, show=True)
