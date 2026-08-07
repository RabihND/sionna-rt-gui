#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
import sys
import time

import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.scene_utils import remove_objects_duplicate_vertices

from . import __version__ as GUI_VERSION
from .analysis import (
    coverage_contents,
    phy_link_contents,
    link_budget_contents,
    link_simulation_contents,
    noise_contents,
    refresh_statistics,
)
from .animation import (
    AnimationConfig,
    animation_gui,
    animation_tick,
    apply_trajectory_position,
    propagate_device_updates,
    restart_trajectories,
)
from .antenna_array import antenna_array_gui
from .assets import (
    ASSET_LIBRARY,
    AssetSpec,
    build_asset_material,
    build_asset_mesh,
)
from sionna.rt.utils.geometry import rotation_matrix
from sionna.rt.utils.render import scene_scale

from . import icons
from .config import (
    DEFAULT_SLICE_PLANE_NAME,
    GuiConfig,
    RadioMapConfig,
    PathsConfig,
    GuiMode,
    RenderingMode,
    RENDERING_MODE_NAMES,
    NVIDIA_GREEN,
)
from .rendering import (
    render_scene,
    set_envmap_rotation,
    add_or_update_ray_traced_image_quantity,
)
from .rm_utils import radio_map_colorbar_to_image
from .ps_utils import (
    ACCENT,
    ACCENT_BRIGHT,
    im_col32,
    set_custom_imgui_style,
    set_polyscope_device_interop_funcs,
    supports_direct_update_from_device,
)
from .sionna_utils import (
    add_paths_to_polyscope,
    add_radio_map_to_polyscope,
    add_scene_to_polyscope,
    get_built_in_scenes,
    get_point_from_camera_center_ray,
    get_normal_for_path,
    set_or_update_radio_devices_polyscope,
)
from .cir_viz import cir_contents, cir_window
from .pattern_viz import (
    pattern_cuts_contents,
    pattern_cuts_window,
    remove_antenna_pattern_structure,
)
from .selection import SelectionType, selection_contents, selection_gui
from .workspace_layout import (
    AreaLayout,
    HEADER_BG,
    area_header,
    begin_area,
    end_area,
    end_property_row,
    property_row,
    set_editor_imgui_style,
    splitter,
    tab_strip,
)

# Editors available in the bottom area
BOTTOM_TABS = ["Timeline", "Impulse response", "Antenna pattern", "Link"]
BOTTOM_TAB_ICONS = {
    "Timeline": icons.timeline_icon,
    "Impulse response": icons.impulse_icon,
    "Antenna pattern": icons.antenna,
    "Link": icons.link_icon,
}

# Tabs of the properties area, in display order
PROPERTIES_TABS = [
    "Object",
    "Devices",
    "Radio map",
    "Paths",
    "Analysis",
    "Assets",
    "Scene",
    "Render",
]
PROPERTIES_TAB_ICONS = {
    "Object": icons.object_icon,
    "Devices": icons.transmitter,
    "Radio map": icons.radio_map,
    "Paths": icons.paths,
    "Analysis": icons.impulse_icon,
    "Assets": icons.car,
    "Scene": icons.scene_icon,
    "Render": icons.render_icon,
}

CTRL_OR_CMD = "Cmd" if sys.platform == "darwin" else "Ctrl"
# Bounding-box outline drawn while picking an object to attach a device to
ATTACH_HIGHLIGHT_NAME = "Attach candidate"
HELP_WINDOW_TABLES = {
    "Camera controls": {
        "Left click + drag": "Camera rotation",
        "Right click + drag": "Camera panning",
        "Shift + left click + drag": "Camera panning",
        "Double left click": "Center camera rotation around clicked point",
        "Mouse scroll": "Camera zoom",
        f"{CTRL_OR_CMD} + shift + left click + drag": "Continuous camera zoom",
        "R": "Reset camera to initial position",
        "F": "Fit scene to camera",
    },
    "Mouse bindings": {
        "Left click": "Select a device, a building or a placed asset",
        "Left click on a radio map": "Probe its value at that point",
        "Right click": "Context menu: add here, select, probe, camera",
        f"{CTRL_OR_CMD} + left click": "Add transmitter at that point",
        f"{CTRL_OR_CMD} + right click": "Add receiver at that point",
        "Drag the gizmo": "Move the selected object",
    },
    "Interface": {
        "Left column": "Add devices and assets, toggle what is shown",
        "Outliner (top right)": "Scene tree: click to select, expand for detail",
        "Properties (right)": "Tabs: object, devices, radio map, paths, analysis",
        "Timeline (bottom)": "Start / pause, speed, and the impulse response",
        f"{CTRL_OR_CMD} + click a number": "Type an exact value instead of dragging",
        "Drag an area border": "Resize the panels",
        "Tab": "Hide or show the whole interface",
    },
    "Key bindings": {
        "H / ?": "Show this help window",
        "K": "Add transmitter at the current mouse position",
        "L": "Add receiver at the current mouse position",
        "M": "Show / hide radio map. Requires at least one transmitter.",
        "C": "Go to next rendering mode",
        "Tab": "Hide or show the interface",
        "Esc": "Close help window or de-select current object",
        "Del (with item selected)": "Delete radio device",
        "Shift + R": "Reload application",
        f"{CTRL_OR_CMD} + Q": "Exit",
    },
    "Slice plane": {
        "S": "Toggle slice plane visibility (not supported in ray-traced rendering mode)",
        "Alt + left click drag": "Move slice plane along its normal",
    },
}


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Sionna RT
        # Scene
        built_in_scenes = get_built_in_scenes()
        self.known_scene_names = ["None"] + list(built_in_scenes.keys())
        self.known_scene_paths = [None] + list(built_in_scenes.values())
        self.current_scene_idx: int = 0
        self.load_scene_requested: str | None = None
        self.scene: rt.Scene | None = None

        # Radio map results
        self.radio_map: rt.RadioMap | None = None
        self.rm_accumulated_samples: int = 0
        self.rm_color_map_options = [v for v in plt.colormaps() if not v.endswith("_r")]
        self.rm_color_map_index = self.rm_color_map_options.index(
            self.cfg.radio_map.color_map
        )
        self.rm_colorbar: np.ndarray | None = None
        self.rm_colorbar_texture_id: int | None = None
        # Due to a current limitation, we can't have alpha on radio maps when using
        # direct updates from the device.
        self.cfg.radio_map.use_alpha = not (
            self.cfg.radio_map.use_direct_update_from_device
            and supports_direct_update_from_device()
        )

        # Paths results
        self.paths: rt.Paths | None = None
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        self.paths_taps: np.ndarray | None = None
        self.paths_cir: tuple[np.ndarray, np.ndarray] | None = None
        # Used to throttle path computations
        self._last_paths_update_time: float = 0.0

        # --- Rendering
        self.render_cache: dict = None
        self.ray_traced_img: mi.TensorXf | None = None
        self.ray_traced_depth: mi.TensorXf | None = None
        self.previous_camera_pose: np.ndarray = ps.get_camera_view_matrix()
        self.rendering_accumulated_samples: int = 0
        self.reset_accumulation_requested: bool = False
        self.denoiser: mi.OptixDenoiser | None = None
        self.slice_plane: ps.SlicePlane | None = None

        # --- Animation state
        self.animation_config: AnimationConfig = AnimationConfig()

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Selections
        self.selected_object: rt.SceneObject | None = None
        self.selected_type: SelectionType | None = None
        self.prev_gizmo_to_world: np.ndarray | None = None
        # Antenna pattern preview (see pattern_viz.py)
        self.show_antenna_pattern: bool = False
        self.antenna_pattern_cache_key: tuple | None = None
        self.antenna_pattern_db_radius: bool = False
        # 0 means "not set yet": a scene-dependent default is picked on first use.
        self.antenna_pattern_scale: float = 0.0
        # Transient "Saved <path>" message: (text, timestamp)
        self.last_export_note: tuple[str, float] | None = None
        # 2D antenna pattern cuts window (see pattern_viz.py)
        self.show_pattern_cuts: bool = False
        self.pattern_cuts_cache: dict | None = None
        # Radio map probe: (cell_i, cell_j, world_position)
        self.rm_probe: tuple[int, int, np.ndarray] | None = None
        # Selected [rx_index, tx_index] pair in the CIR window
        self.cir_pair: list[int] = [0, 0]
        # CIR delay axis: auto-scale, or fixed maximum (0 = pick a
        # scene-dependent default on first use)
        self.cir_auto_delay: bool = True
        self.cir_max_delay_ns: float = 0.0
        # CIR level axis (None = seed from the data on first use)
        self.cir_auto_level: bool = True
        self.cir_max_db: float | None = None
        # Devices pointed at a target: device name -> target name, plus the
        # positions each was last aimed from and at
        self.look_at_targets: dict[str, str] = {}
        self._look_at_state: dict[str, tuple] = {}
        # Devices attached to scene objects: device name ->
        # {"object": object name, "offset": device minus object position}
        self.attachments: dict[str, dict] = {}
        # Lazily-built full traversal of the wireless scene (see
        # update_attached_object); reset on scene load.
        self._full_scene_params = None
        # Device name waiting for the user to click a scene object to
        # attach to (one-shot picking mode), and the object currently
        # under the cursor while picking.
        self.attach_pick_pending: str | None = None
        self.attach_hover_name: str | None = None
        # Mesh tinted for the hover highlight: (mesh id, original color)
        self._attach_tinted: tuple[str, tuple] | None = None
        # Rendering mode to restore when object-picking ends
        self._attach_previous_rendering_mode: RenderingMode | None = None
        # Names of assets placed from the library (see assets.py)
        self.placed_assets: list[str] = []
        # Viewer colour per placed asset, since materials are shared
        self.asset_colors: dict[str, tuple] = {}
        # Ground-like objects that clicks pass through
        self.background_objects: set[str] = set()
        # Cached traversal of the ray-traced scene, and the scene it belongs to
        self._visual_params = None
        self._visual_params_of = None
        self._visual_vertex_keys: dict[str, str] = {}
        # Sizes of the docked areas (see workspace_layout.py)
        self.layout: AreaLayout = AreaLayout()
        # Index of the visible properties tab, and of the bottom editor
        self.properties_tab: int = 0
        self.bottom_tab: int = 0
        # Cached analysis readouts (see analysis.py)
        self._statistics_time: float = 0.0
        self.radio_map_stats: dict | None = None
        self.link_budget_stats: dict | None = None
        self.link_simulation_stats: dict | None = None
        # Link metrics from sionna's system-level layer, measured on demand
        self.phy_metrics_stats: dict | None = None
        self.phy_bler_target: float = 0.1
        self.phy_auto_measure: bool = True
        self.nr_link_stats: dict | None = None
        # When the link was last measured, to tell whether the channel has
        # changed underneath the results
        self.link_metrics_time: float = 0.0
        self.coverage_threshold_dbm: float = -95.0
        # Solver work per frame, steered by the measured frame time
        self._rm_refine_samples: int = 0
        self._baseline_frame_s: float | None = None
        self.solver_update_delay_s: float = 0.0
        self.solver_delay_max_s: float = 0.12
        # State of the settle pass that follows interactive tracing
        self._last_interactive_paths: float = 0.0
        self._settle_paths_pending: bool = False
        self.frame_time_target_s: float = 1.0 / 30.0
        # The simulation only runs once started, and can then be paused
        self.simulation_running: bool = False
        self.simulation_started: bool = False
        self.compute_radio_map_on_start: bool = False
        # Previous gizmo transform while moving a scene object
        self.object_gizmo_previous: np.ndarray | None = None
        # Right-click context menu state
        self.context_menu_requested: bool = False
        self.context_menu_position: np.ndarray | None = None
        self.context_menu_object: str | None = None

        # --- Polyscope setup
        # Can be used to derive e.g. random seeds.
        self.ps_groups: dict[str, ps.Group] = {}
        self.frame_i: int = 0
        self.was_mouse_dragging: bool = False
        self.home_camera_to_world: np.ndarray | None = None

        # Pre-init settings
        ps.set_program_name(self.cfg.title)
        # Window size and position will be loaded from the last run from `.polyscope.ini`
        ps.set_use_prefs_file(True)
        ps.set_enable_vsync(cfg.use_vsync)
        # On some machines (especially with remote access), VSync doesn't do anything,
        # so we cap fps as well.
        ps.set_max_fps(60)
        ps.set_user_gui_is_on_right_side(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_verbosity(1)
        ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)
        ps.set_background_color(self.cfg.background_color)
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_give_focus_on_show(True)
        ps.set_transparency_mode("pretty")
        ps.set_files_dropped_callback(self.on_files_dropped)
        if self.cfg.use_docked_layout:
            set_editor_imgui_style()
        else:
            set_custom_imgui_style()

        was_initialized = ps.is_initialized()
        if not was_initialized:
            ps.set_up_dir("z_up")
            ps.set_front_dir("y_front")
            ps.init()
            if supports_direct_update_from_device():
                set_polyscope_device_interop_funcs()

        # Set window size once UI scale is known (after init).
        self.ui_scale: float = ps.get_ui_scale()
        ps.set_window_size(
            self.cfg.rendering.default_resolution[0] * self.ui_scale,
            self.cfg.rendering.default_resolution[1] * self.ui_scale,
        )
        # Used to track window size changes. Includes DPI scaling.
        self.previous_window_resolution: tuple[int, int] = ps.get_window_size()
        # Note that our `set_window_size()` call may not succeed, e.g. if the window is
        # maximized. We adopt the effective resolution to make sure that we rendering with
        # the right aspect ratio, etc.
        self.cfg.rendering.current_resolution = tuple(
            v // self.ui_scale for v in self.previous_window_resolution
        )
        # Enable denoiser if requested.
        if "cuda" in mi.variant():
            self.set_use_denoiser(self.cfg.rendering.use_denoiser)
        else:
            # If there is no CUDA-compatible GPU, switch to rasterization (cheaper).
            self.cfg.rendering.mode = RenderingMode.RASTERIZATION

        # Used to throttle window size change handling. Set to None if no change is pending.
        self.last_window_size_changed_time: float | None = None

        # --- Load scene
        # TODO: preserve currently-selected scene across reloads
        if not self.cfg.scene_filename.endswith(".xml"):
            if self.cfg.scene_filename not in built_in_scenes:
                raise ValueError(
                    f'Scene "{self.cfg.scene_filename}" not found. Built-in scenes: {list(built_in_scenes.keys())}'
                )
            self.cfg.scene_filename = built_in_scenes[self.cfg.scene_filename]
        self.load_scene(
            self.cfg.scene_filename,
            # If the program was just reloaded (live coding), don't move the camera
            recenter_camera=not was_initialized,
        )

        # --- Slice plane
        ps.remove_all_slice_planes()
        plane = ps.add_scene_slice_plane()
        plane.set_pose(
            (
                self.cfg.rendering.slice_plane_position
                if self.cfg.rendering.slice_plane_position is not None
                else (0, 0, self.scene.mi_scene.bbox().center().z)
            ),
            self.cfg.rendering.slice_plane_normal,
        )
        plane.set_active(
            self.cfg.rendering.default_slice_plane_enabled
            and (self.cfg.rendering.mode == RenderingMode.RASTERIZATION)
        )
        self.slice_plane = plane

        # --- Example scenario
        if self.cfg.create_example_scenario:
            self.create_example_scenario(
                set_camera=not was_initialized, add_radio_map=False
            )

    def create_example_scenario(
        self, set_camera: bool = True, add_radio_map: bool = True
    ):
        if set_camera:
            self.home_camera_to_world = np.array(
                [
                    [
                        2.0079615e-03,
                        -9.9999154e-01,
                        -3.9256822e-08,
                        4.4776478e00,
                    ],
                    [7.8317523e-01, 1.5742097e-03, 6.2179816e-01, 1.1707677e01],
                    [
                        -6.2179959e-01,
                        -1.2489425e-03,
                        7.8317869e-01,
                        -2.2836572e02,
                    ],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            )
            self.move_camera_home()

        # Add some example transmitters
        for pos in [
            [-34.0, 13.0, 33.0],
        ]:
            self.add_radio_device(
                pos, is_transmitter=True, allow_auto_update=False, select=False
            )

            shifted = [pos[0] + 15, pos[1] - 14, pos[2] - 20]
            self.add_radio_device(
                shifted, is_transmitter=False, allow_auto_update=False, select=False
            )

        # Example animation
        p = self.scene.get("rx-0").position.numpy().squeeze()
        traj = self.animation_config.trajectories["rx-0"]
        traj.add_point(p - [0, 30, 0])
        traj.add_point(p)
        traj.add_point(p + [40, 0, 0])
        traj.enabled = True
        traj.distance = 0.0  # Start at the first point
        self.animation_config.speed_multiplier = 10.0
        # The scenario is only set up here: computing and animating waits for
        # the user to press Start.
        self.animation_config.playing = False
        self.compute_radio_map_on_start = add_radio_map

    def reset_and_setup_structures(self):
        # Clear Sionna state
        self.clear_radio_map()
        self.clear_selection()
        self.clear_paths()
        self.clear_ray_traced_image()
        self.animation_config.clear()
        # Scene-dependent defaults are re-picked on first use
        self.antenna_pattern_scale = 0.0
        self.cir_max_delay_ns = 0.0
        self.attachments.clear()
        self._full_scene_params = None
        self.attach_pick_pending = None
        self.attach_hover_name = None
        # The tinted mesh is gone with the old scene
        self._attach_tinted = None
        self._attach_previous_rendering_mode = None
        self.placed_assets.clear()
        self.asset_colors.clear()

        # Clear Polyscope state
        ps.remove_all_structures()
        ps.remove_all_groups()

        self.ps_groups = {
            "scene": ps.create_group("Scene meshes"),
            "rd": ps.create_group("Radio devices"),
            "radio_maps": ps.create_group("Radio maps"),
            "paths": ps.create_group("Paths"),
        }

    def load_scene(self, scene_path: str, recenter_camera: bool = True):
        self.reset_and_setup_structures()
        self.scene = rt.load_scene(scene_path)
        remove_objects_duplicate_vertices(self.scene.mi_scene)

        self.scene.tx_array = self.cfg.tx_array.create()
        self.scene.rx_array = self.cfg.rx_array.create()

        thickness = self.cfg.radio_material_thickness
        scattering_coefficient = self.cfg.radio_material_scattering_coefficient
        for sh in self.scene.mi_scene.shapes():
            bsdf = sh.bsdf()
            if isinstance(bsdf, rt.RadioMaterialBase):
                if thickness is not None:
                    bsdf.thickness = thickness
                if scattering_coefficient is not None:
                    bsdf.scattering_coefficient = scattering_coefficient

        self.scene.bandwidth = self.cfg.bandwidth_hz
        self.cfg.scene_filename = scene_path

        # Scene statistics shown in the GUI, computed once per scene load
        shapes = self.scene.mi_scene.shapes()
        n_triangles = sum(sh.face_count() for sh in shapes)
        self.scene_stats = (len(shapes), n_triangles)

        # Ground-like objects: nearly flat and covering most of the scene.
        # Clicks pass through them, since selecting the ground is never useful.
        scene_extents = np.array(self.scene.mi_scene.bbox().extents())
        scene_footprint = max(scene_extents[0] * scene_extents[1], 1e-6)
        self.background_objects = set()
        for name, scene_object in self.scene.objects.items():
            extents = np.array(scene_object.mi_mesh.bbox().extents())
            footprint = extents[0] * extents[1]
            is_flat = extents[2] <= 0.02 * max(extents[0], extents[1])
            if is_flat and footprint >= 0.4 * scene_footprint:
                self.background_objects.add(name)

        # Add this scene to the list of known scene names, if missing
        if scene_path not in self.known_scene_paths:
            self.known_scene_names.append(scene_path)
            self.known_scene_paths.append(scene_path)

        try:
            self.current_scene_idx = self.known_scene_paths.index(
                self.cfg.scene_filename
            )
        except ValueError:
            self.current_scene_idx = 0

        add_scene_to_polyscope(self.scene, self.ps_groups)
        self.set_rendering_mode(self.cfg.rendering.mode)
        if recenter_camera:
            self.fit_camera_to_scene()

        if (self.cfg.rendering.slice_plane_position is None) and (
            self.slice_plane is not None
        ):
            self.slice_plane.set_pose(
                (0, 0, self.scene.mi_scene.bbox().center().z),
                self.cfg.rendering.slice_plane_normal,
            )

    def on_files_dropped(self, files: list[str]):
        for file in files:
            if file.endswith(".xml"):
                print(f"[i] Loading dropped XML file: {file}")
                try:
                    self.load_scene(file)
                except Exception as e:
                    print(f'[!] Failed loading scene "{file}":\n{e}')
                    continue
                # Only handle one valid XML file
                break

    def move_camera_home(self):
        """Move camera to the home view, if any. Otherwise, fits the scene in the camera viewport."""
        if self.home_camera_to_world is not None:
            # We probably can't rely on ps.pick() since the buffers probably haven't been
            # updated yet with the new camera pose. So we trace our own ray to get the rotation
            # point of the turntable navigation.
            if ps.get_navigation_style() == "turntable":
                center = get_point_from_camera_center_ray(
                    self.scene, np.linalg.inv(self.home_camera_to_world)
                )
                if center is not None:
                    ps.set_view_center_raw(center)

            ps.set_camera_view_matrix(self.home_camera_to_world)

            # The camera pose snapshot used to detect camera motion is taken
            # at the end of the tick, after this change: request an explicit
            # re-render, otherwise the ray-traced image keeps the old pose.
            self.reset_accumulation_requested = True
            ps.request_redraw()
        else:
            self.fit_camera_to_scene()

    def move_camera_top(self):
        """Bird's-eye view: look straight down at the scene from above."""
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        bbox = self.scene.mi_scene.bbox()
        center = np.array(bbox.center())
        extents = np.array(bbox.extents())

        footprint = max(extents[0], extents[1], 1.0)
        distance = footprint / (
            2.0 * np.tan(0.5 * np.radians(fov_vertical_deg))
        ) + 0.5 * extents[2]
        # Looking exactly down the up axis is degenerate (Polyscope warns and
        # rejects the view), so keep a small but safe tilt.
        eye = center + np.array([0.0, -0.05 * distance, distance])
        ps.look_at(eye, center, fly_to=True)
        if ps.get_navigation_style() == "turntable":
            ps.set_view_center_raw(center)
        self.reset_accumulation_requested = True
        ps.request_redraw()

    def fit_camera_to_scene(self):
        """Move the camera to a position where most of the scene is visible."""
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        bbox = self.scene.mi_scene.bbox()

        center = bbox.center()
        extents = bbox.extents()

        # Calculate the required distance to fit the scene in the vertical FOV
        # We use the larger of height or diagonal extent for safety margin
        scene_diagonal = np.linalg.norm(extents)
        scene_size = max(extents.z, scene_diagonal * 0.8)  # 0.8 for some margin

        # Convert FOV to radians and calculate required distance
        fov_vertical_rad = np.radians(fov_vertical_deg)
        distance = scene_size / (2.0 * np.tan(0.5 * fov_vertical_rad))
        distance = distance * 0.95

        # Position camera at a reasonable viewing angle.
        # Angle above horizontal
        es, ec = dr.sincos(np.radians(45))
        # Angle from positive X axis
        zs, zc = dr.sincos(np.radians(0))
        origin = [
            center.x + distance * ec * zc,
            center.y + distance * ec * zs,
            center.z + distance * es,
        ]
        target = [center.x - 0.2 * (center.x - origin[0]), center.y, center.z]

        ps.set_view_center_raw(target)
        ps.look_at(origin, target)
        # See move_camera_home: force the ray-traced view to follow the jump.
        self.reset_accumulation_requested = True

    # ------------------------

    def tick(self):
        if self.load_scene_requested is not None:
            try:
                self.load_scene(self.load_scene_requested)
            except Exception as e:
                print(f'[!] Failed loading scene "{self.load_scene_requested}":\n{e}')
            self.load_scene_requested = None

        self.process_inputs()

        # --- Resolution changes
        current_window_resolution = ps.get_window_size()
        now = time.time()
        if current_window_resolution != self.previous_window_resolution:
            self.last_window_size_changed_time = now
            self.previous_window_resolution = current_window_resolution
        if (self.last_window_size_changed_time is not None) and (
            now - self.last_window_size_changed_time > 0.3
        ):
            self.clear_ray_traced_image()
            self.set_rendering_resolution(current_window_resolution)
            self.last_window_size_changed_time = None

        # --- Rendering
        if self.cfg.rendering.mode == RenderingMode.RAY_TRACING:
            camera_changed = self.reset_accumulation_requested or not np.allclose(
                self.previous_camera_pose, ps.get_camera_view_matrix()
            )
            if camera_changed:
                self.rendering_reset_accumulation()
                self.reset_accumulation_requested = False

            if (
                self.rendering_accumulated_samples
                < self.cfg.rendering.max_accumulated_spp
            ):
                # TODO: we could potentially skip rendering depth in subsequent frames,
                #       since we only accumulate RGB.
                new_img, aovs, self.render_cache = render_scene(
                    self.cfg.rendering,
                    self.scene,
                    seed=self.frame_i,
                    camera_changed=camera_changed,
                    cache=self.render_cache,
                    use_denoiser=self.denoiser is not None,
                )
                if self.ray_traced_img is None:
                    self.ray_traced_img = new_img
                    self.ray_traced_depth = aovs[0]
                else:
                    t = self.cfg.rendering.spp_per_frame / (
                        self.rendering_accumulated_samples
                        + self.cfg.rendering.spp_per_frame
                    )
                    t = dr.opaque(mi.Float32, t)
                    self.ray_traced_img = (1 - t) * self.ray_traced_img + t * new_img
                    # Keep using 1spp depth, it looks better than accumulating.
                    self.ray_traced_depth = aovs[0]

                self.rendering_accumulated_samples += self.cfg.rendering.spp_per_frame

                if self.denoiser is not None:
                    to_sensor = self.render_cache["sensor"].world_transform().inverse()
                    self.ray_traced_img = self.denoiser(
                        self.ray_traced_img,
                        albedo=aovs[1],
                        normals=aovs[2],
                        to_sensor=to_sensor,
                    )

                add_or_update_ray_traced_image_quantity(
                    self.ray_traced_img, self.ray_traced_depth
                )

        self.observe_frame_time(psim.GetIO().DeltaTime)

        # Once movement stops, replace the interactive trace with a full one
        if (
            self._settle_paths_pending
            and time.time() - self._last_interactive_paths > 0.35
        ):
            self._settle_paths_pending = False
            self._last_paths_update_time = 0.0  # bypass the throttle for this one
            self.update_paths(show=True)

        # --- Automatic refinement of the radio map
        if self.simulation_running and self.radio_map is not None:
            if (
                self.rm_accumulated_samples
                < self.cfg.radio_map.accumulate_max_samples_per_tx
            ):
                # Refine with as many samples as fit the frame budget, instead
                # of a full solve per frame: the map converges just as fast in
                # wall-clock terms while the interface stays responsive.
                samples_per_tx = self.rm_refine_samples_per_tx()
                self._rm_refine_samples = samples_per_tx
                rm_new = self.compute_radio_map(samples_per_tx=samples_per_tx)
                if rm_new is not None:
                    self.radio_map._pathgain_map += rm_new.path_gain
                    self.rm_accumulated_samples += samples_per_tx
                    # Note: vmin, vmax didn't change so we don't update the colorbar.
                    add_radio_map_to_polyscope(
                        "radio_map",
                        self.radio_map,
                        self.ps_groups,
                        self.cfg.radio_map,
                        direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                        use_alpha=self.cfg.radio_map.use_alpha,
                    )

        # --- Radio device animations
        if self.simulation_running:
            animation_tick(self, psim.GetIO().DeltaTime)

        # --- Devices that are locked onto a target follow it
        self.apply_look_at_targets()

        # --- GUI
        self.gui()
        self.frame_i += 1
        self.previous_camera_pose = ps.get_camera_view_matrix()

    # ------------------------

    def set_rendering_mode(self, mode: RenderingMode):
        self.cfg.rendering.mode = mode
        is_ray_tracing = self.cfg.rendering.mode == RenderingMode.RAY_TRACING

        if is_ray_tracing and (self.slice_plane is not None):
            self.slice_plane.set_active(False)

        if not is_ray_tracing:
            self.clear_ray_traced_image()

        # Hide Polyscope-side meshes if we are ray tracing.
        self.ps_groups["scene"].set_enabled(not is_ray_tracing)

    def set_use_denoiser(self, use_denoiser: bool):
        self.cfg.rendering.use_denoiser = use_denoiser

        # The integrator will be re-created, so we reset the rendering cache.
        self.render_cache = None

        if self.cfg.rendering.use_denoiser:
            self.denoiser = mi.OptixDenoiser(
                input_size=self.cfg.rendering.rendering_resolution,
                albedo=True,
                normals=True,
                temporal=False,
            )
        else:
            self.denoiser = None
        self.rendering_reset_accumulation()

    def rendering_reset_accumulation(self):
        self.rendering_accumulated_samples = 0
        if self.ray_traced_img is not None:
            self.ray_traced_img[:] = 0
            self.ray_traced_depth[:] = 0

    def clear_ray_traced_image(self):
        import polyscope_bindings as psb

        self.render_cache = None
        self.ray_traced_img = None
        self.ray_traced_depth = None
        self.rendering_accumulated_samples = 0
        psb.get_global_floating_quantity_structure().remove_quantity(
            "ray_traced_img", errorIfAbsent=False
        )

    def set_rendering_resolution(self, window_resolution: tuple[int, int]):
        self.cfg.rendering.current_resolution = tuple(
            v // self.ui_scale for v in window_resolution
        )
        self.clear_ray_traced_image()
        # Re-create denoiser for the new resolution, if appropriate
        self.set_use_denoiser(self.cfg.rendering.use_denoiser)

    # ------------------------

    def set_radio_map(self, radio_map: rt.RadioMap, show: bool = False):
        self.radio_map = radio_map
        self.rm_accumulated_samples = 0
        self.update_radio_map_colorbar()

        if show:
            add_radio_map_to_polyscope(
                "radio_map",
                self.radio_map,
                self.ps_groups,
                self.cfg.radio_map,
                direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                use_alpha=self.cfg.radio_map.use_alpha,
            )

    def update_radio_map_colorbar(self):
        # Draw the colorbar to an array.
        self.rm_colorbar = radio_map_colorbar_to_image(
            self.cfg.radio_map.color_map,
            self.cfg.radio_map.vmin,
            self.cfg.radio_map.vmax,
        )
        # Note: we upload the colorbar to a texture, even if it's not shown right now.
        ps.add_color_alpha_image_quantity(
            "rm_colorbar",
            values=self.rm_colorbar,
            enabled=True,
            image_origin="upper_left",
            show_fullscreen=False,
            # show_in_camera_billboard=True,
        )
        self.rm_colorbar_texture_id = ps.get_quantity_buffer(
            "rm_colorbar", "colors"
        ).get_texture_native_id()

    def compute_radio_map(self, samples_per_tx: int | None = None) -> rt.RadioMap | None:
        if not self.scene._transmitters:
            return None

        solver = rt.RadioMapSolver()
        if samples_per_tx is None:
            samples_per_tx = self.cfg.radio_map.samples_per_it // len(
                self.scene._transmitters
            )
        samples_per_tx = max(int(samples_per_tx), 1024)
        return solver(
            self.scene,
            seed=self.frame_i,
            center=self.cfg.radio_map.center,
            orientation=self.cfg.radio_map.orientation,
            size=self.cfg.radio_map.size,
            cell_size=self.cfg.radio_map.cell_size,
            measurement_surface=self.cfg.radio_map.measurement_surface,
            # precoding_vec=self.cfg.radio_map.precoding_vec,
            samples_per_tx=samples_per_tx,
            rr_depth=self.cfg.radio_map.rr_depth,
            rr_prob=self.cfg.radio_map.rr_prob,
            stop_threshold=(
                10.0 ** (self.cfg.radio_map.stop_threshold_db / 10.0)
                if self.cfg.radio_map.stop_threshold_db is not None
                else None
            ),
            max_depth=self.cfg.radio_map.max_depth,
            los=self.cfg.radio_map.los,
            specular_reflection=self.cfg.radio_map.specular_reflection,
            diffuse_reflection=self.cfg.radio_map.diffuse_reflection,
            refraction=self.cfg.radio_map.refraction,
            diffraction=self.cfg.radio_map.diffraction,
            edge_diffraction=self.cfg.radio_map.edge_diffraction,
            diffraction_lit_region=self.cfg.radio_map.diffraction_lit_region,
        )

    def observe_frame_time(self, frame_time_s: float) -> None:
        """
        Steer how much solver work each frame takes, from the frame time itself.

        Timing the solvers directly would mean synchronising with the GPU, which
        destroys the overlap Dr.Jit relies on and makes everything slower, so the
        frame time is the signal instead: too slow, ask for less work; comfortably
        fast, ask for more.
        """
        if frame_time_s <= 0.0:
            return

        # The cheapest recent frame approximates what drawing alone costs. It
        # decays upwards slowly, so a heavier scene or a bigger window is picked
        # up rather than held against the solvers forever.
        if self._baseline_frame_s is None:
            self._baseline_frame_s = frame_time_s
        else:
            self._baseline_frame_s = min(
                frame_time_s, self._baseline_frame_s * 1.01
            )

        # Only the time above what drawing costs is the solvers' to give back.
        # Without this the controller starves them whenever rendering alone is
        # slower than the target, and the radio map stops converging.
        target = max(self.frame_time_target_s, self._baseline_frame_s * 1.25)

        ceiling = float(max(self.cfg.radio_map.samples_per_it, 1024))
        floor = float(min(max(ceiling / 64.0, 10_000.0), ceiling))
        if frame_time_s > target:
            # Back off quickly
            self._rm_refine_samples = int(
                min(max(self._rm_refine_samples * 0.6, floor), ceiling)
            )
            # Paths are a live link between devices, so a long wait reads as
            # lag rather than smoothness: never wait longer than the user allows.
            self.solver_update_delay_s = min(
                max(self.solver_update_delay_s * 1.5, 0.03), self.solver_delay_max_s
            )
        elif frame_time_s < 0.85 * target:
            # Creep back up
            self._rm_refine_samples = int(
                min(max(self._rm_refine_samples * 1.2, floor), ceiling)
            )
            self.solver_update_delay_s = max(self.solver_update_delay_s * 0.8, 0.0)

    def link_budget_tx_power_dbm(self, tx_index: int = 0) -> float:
        """Transmit power of a transmitter by index, in dBm."""
        transmitters = list(self.scene._transmitters.values())
        if tx_index >= len(transmitters):
            return 0.0
        return float(transmitters[tx_index].power_dbm[0])

    def rm_refine_samples_per_tx(self) -> int:
        """Samples per transmitter for one refinement step, within the budget."""
        configured = max(
            self.cfg.radio_map.samples_per_it // max(len(self.scene._transmitters), 1),
            1024,
        )
        if self._rm_refine_samples <= 0:
            # Start modestly; observe_frame_time grows this if there is headroom
            self._rm_refine_samples = min(configured, 500_000)
        self._rm_refine_samples = max(
            self._rm_refine_samples, min(configured // 64, configured)
        )
        return int(min(self._rm_refine_samples, configured))

    def has_visible_radio_map(self) -> tuple[bool, ps.SurfaceMesh]:
        rm_struct = None
        if ps.has_surface_mesh("radio_map"):
            rm_struct = ps.get_surface_mesh("radio_map")
            rm_visible = rm_struct.is_enabled()
        else:
            rm_visible = False

        return rm_visible, rm_struct

    # ------------------------

    def update_paths(
        self, clear_first: bool = False, show: bool = True, interactive: bool = False
    ):
        # Throttle path computations: never spend more than a third of the time
        # solving paths, using what the last solve actually cost.
        current_time = time.time()
        time_since_last_update = current_time - self._last_paths_update_time
        minimum_delay = max(
            self.cfg.paths.min_update_delay_s, self.solver_update_delay_s
        )
        if time_since_last_update < minimum_delay:
            # Skip this update
            return

        if clear_first:
            self.clear_paths()

        self.paths = self.compute_paths(interactive=interactive)
        self._last_paths_update_time = time.time()
        if interactive:
            # Something is moving: trace again at full quality once it settles
            self._last_interactive_paths = self._last_paths_update_time
            self._settle_paths_pending = True
        else:
            self._settle_paths_pending = False
        if self.paths is None:
            return

        if show:
            add_paths_to_polyscope(self, self.paths, self.ps_groups)

        if self.cfg.paths.compute_cir:
            self.paths_taps = self.paths.taps(
                bandwidth=self.cfg.paths.bandwidth,
                l_min=self.cfg.paths.l_min,
                l_max=self.cfg.paths.l_max,
                sampling_frequency=self.cfg.paths.sampling_frequency,
                num_time_steps=self.cfg.paths.num_time_steps,
                normalize=self.cfg.paths.normalize,
                normalize_delays=self.cfg.paths.normalize_delays,
                out_type="numpy",
            )
            self.paths_cir = self.paths.cir(normalize_delays=True, out_type="numpy")

            # If self.paths_cir[0] has no elements, replace self.paths_taps with zeros, first element = 1e-10
            if np.sum(self.paths_cir[0]) == 0:
                # Set the first element to 1e-10
                self.paths_taps[0, 0, 0, 0, 0, 0] = 1e-10

    def compute_paths(self, interactive: bool = False) -> rt.Paths | None:
        if not self.scene._transmitters or not self.scene._receivers:
            return None

        # While things move, trace with fewer samples. Measurements on the
        # example scene show the same total path gain either way: the dominant
        # paths are found deterministically and extra samples only add weak
        # diffuse ones, so movement can be followed far more closely for it.
        samples_per_src = self.cfg.paths.samples_per_src
        if interactive:
            samples_per_src = max(
                min(self.cfg.paths.interactive_samples_per_src, samples_per_src), 1000
            )

        solver = rt.PathSolver()
        return solver(
            self.scene,
            max_depth=self.cfg.paths.max_depth,
            max_num_paths_per_src=self.cfg.paths.max_num_paths_per_src,
            samples_per_src=samples_per_src,
            synthetic_array=self.cfg.paths.synthetic_array,
            los=self.cfg.paths.los,
            specular_reflection=self.cfg.paths.specular_reflection,
            diffuse_reflection=self.cfg.paths.diffuse_reflection,
            refraction=self.cfg.paths.refraction,
            diffraction=self.cfg.paths.diffraction,
            edge_diffraction=self.cfg.paths.edge_diffraction,
            diffraction_lit_region=self.cfg.paths.diffraction_lit_region,
            seed=self.cfg.paths.seed,
        )

    # ------------------------

    def default_new_device_position(self, is_transmitter: bool) -> list[float]:
        """
        Pick a sensible default position for a device added from the GUI:
        near the center of the scene, up high, with a small offset per
        existing device to avoid stacking them on top of each other.
        """
        bbox = self.scene.mi_scene.bbox()
        center = 0.5 * (bbox.min + bbox.max)
        extents = bbox.max - bbox.min
        count = len(
            self.scene._transmitters if is_transmitter else self.scene._receivers
        )
        # Transmitters spawn slightly to one side, receivers to the other.
        side = 1.0 if is_transmitter else -1.0
        spacing = max(0.02 * extents.x, 2.0)
        return [
            center.x + side * (0.1 * extents.x + spacing * count),
            center.y,
            center.z + 0.35 * extents.z,
        ]

    def add_radio_device(
        self,
        position: list[float],
        is_transmitter: bool,
        allow_auto_update: bool = True,
        select: bool = True,
    ) -> rt.RadioDevice:
        # TODO: controllable offset to the clicked surface (along normal?)
        existing_rd = (
            self.scene._transmitters if is_transmitter else self.scene._receivers
        )

        # Add actual radio device to Sionna scene
        prefix = "tx" if is_transmitter else "rx"
        free_index = len(existing_rd)
        while f"{prefix}-{free_index}" in existing_rd:
            free_index += 1

        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{prefix}-{free_index}",
            position=position,
            orientation=[0, 0, 0],
        )
        if is_transmitter:
            new_rd.power_dbm = self.cfg.default_tx_power_dbm
        self.scene.add(new_rd)

        set_or_update_radio_devices_polyscope(
            existing_rd,
            is_transmitter,
            self,
        )

        if select:
            # Select the new device right away so it can be moved with the
            # gizmo without an extra click.
            self.selected_object = new_rd
            self.selected_type = (
                SelectionType.Transmitter if is_transmitter else SelectionType.Receiver
            )

        if (
            allow_auto_update
            and self.cfg.radio_map.auto_update
            and is_transmitter
            and (self.radio_map is not None)
        ):
            self.set_radio_map(self.compute_radio_map(), show=True)
        if allow_auto_update and self.cfg.paths.auto_update:
            self.update_paths(show=True)

        return new_rd

    def remove_object(
        self, object: rt.SceneObject, selected_type: SelectionType
    ) -> None:
        match selected_type:
            case SelectionType.Transmitter:
                del self.scene._transmitters[object.name]
            case SelectionType.Receiver:
                del self.scene._receivers[object.name]
            case _:
                print(f"[!] Unexpected selection type: {selected_type}")
                pass

        if object.name in self.animation_config.trajectories:
            del self.animation_config.trajectories[object.name]
        self.attachments.pop(object.name, None)
        self.look_at_targets.pop(object.name, None)
        self._look_at_state.pop(object.name, None)
        # Anything that was pointed at it stops tracking
        for name, target in list(self.look_at_targets.items()):
            if target == object.name:
                self.look_at_targets.pop(name, None)

    def attach_device_to_object(
        self,
        device: rt.RadioDevice,
        object_name: str | None,
        snap_position: np.ndarray | None = None,
    ) -> None:
        """
        Attach a radio device to a scene object: the device snaps on top of
        the object (or just above `snap_position`, e.g. the clicked point),
        and from then on the object follows the device's motion
        (trajectories, gizmo, position edits). Pass None to detach.
        """
        if object_name is None:
            self.attachments.pop(device.name, None)
            return

        scene_object = self.scene.get(object_name)
        if scene_object is None:
            return
        object_position = scene_object.position.numpy().squeeze()
        if snap_position is not None:
            new_position = np.asarray(snap_position, dtype=float) + [0.0, 0.0, 1.5]
        else:
            top_z = float(scene_object.mi_mesh.bbox().max.z)
            new_position = np.array(
                [object_position[0], object_position[1], top_z + 1.0]
            )
        device.position = mi.Point3f(*new_position.tolist())
        dr.make_opaque(device.position)
        self.attachments[device.name] = {
            "object": object_name,
            "offset": device.position.numpy().squeeze() - object_position,
            # Device rotation at attach time, and the rotation applied to the
            # object so far, both relative to that moment (see
            # update_attached_object).
            "device_rotation_0": np.array(
                rotation_matrix(device.orientation).numpy()
            ).squeeze(),
            "applied_rotation": np.eye(3),
        }

    def _visual_scene_vertex_key(self, mesh_id: str) -> str | None:
        """
        Key of a mesh's vertex buffer in the ray-traced scene. That scene is
        built from the wireless one and prefixes shape names with their index
        ("shape-6-car-1"), so the plain mesh id does not match.
        """
        if self.render_cache is None:
            return None
        visual_scene = self.render_cache["visual_scene"]
        if self._visual_params_of is not visual_scene:
            self._visual_params = mi.traverse(visual_scene)
            self._visual_params_of = visual_scene
            self._visual_vertex_keys = {}
            suffix = ".vertex_positions"
            for key in self._visual_params.keys():
                if not key.endswith(suffix):
                    continue
                shape_name = key[: -len(suffix)]
                parts = shape_name.split("-", 2)
                base = (
                    parts[2] if len(parts) == 3 and parts[0] == "shape" else shape_name
                )
                self._visual_vertex_keys[base] = key
        return self._visual_vertex_keys.get(mesh_id)

    def _update_ray_traced_geometry(self, mesh) -> None:
        """
        Move a mesh in the ray-traced scene as well. Without this the ray-traced
        view keeps drawing the object at its old place, which looks like a
        duplicate that cannot be clicked.
        """
        if self.render_cache is None:
            return
        key = self._visual_scene_vertex_key(mesh.id())
        if key is None:
            # Not in the ray-traced scene yet: rebuild it on the next frame
            self.render_cache = None
            return
        try:
            self._visual_params[key] = mesh.vertex_positions_buffer()
            self._visual_params.update()
        except Exception as e:
            logging.debug("Could not update the ray-traced scene: %s", e)
            self.render_cache = None

    def _after_geometry_edit(self, scene_object: rt.SceneObject) -> None:
        """Refresh the views after an object's geometry moved."""
        mesh = scene_object.mi_mesh
        if ps.has_surface_mesh(mesh.id()):
            ps.get_surface_mesh(mesh.id()).update_vertex_positions(
                mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            )
        self._update_ray_traced_geometry(mesh)
        self.reset_accumulation_requested = True

    def transform_scene_object(
        self,
        scene_object: rt.SceneObject,
        translation=None,
        rotation_increment=None,
    ) -> bool:
        """
        Translate and / or rotate a scene object by editing its vertices in the
        wireless scene. Used for objects whose mesh buffers sionna's cached
        scene parameters do not expose.
        """
        needs_translation = translation is not None and not np.allclose(
            translation, 0.0, atol=1e-6
        )
        needs_rotation = rotation_increment is not None and not np.allclose(
            rotation_increment, np.eye(3), atol=1e-6
        )
        if not needs_translation and not needs_rotation:
            return False

        mesh = scene_object.mi_mesh
        if self._full_scene_params is None:
            self._full_scene_params = mi.traverse(self.scene.mi_scene)
        key = mesh.id() + ".vertex_positions"
        if key not in self._full_scene_params:
            logging.warning("Cannot move scene object %s: no %s", mesh.id(), key)
            return False

        vertices = dr.unravel(mi.Point3f, self._full_scene_params[key])
        if needs_rotation:
            # Rotate around the object's own center, then translate
            center = mi.Point3f(*scene_object.position.numpy().squeeze().tolist())
            vertices = (
                mi.Matrix3f(np.asarray(rotation_increment).tolist())
                @ (vertices - center)
            ) + center
        if needs_translation:
            vertices = vertices + mi.Point3f(
                *np.asarray(translation, dtype=float).tolist()
            )
        self._full_scene_params[key] = dr.ravel(vertices)
        self._full_scene_params.update()
        # Invalidate sionna's solver caches
        self.scene.scene_geometry_updated()
        self._after_geometry_edit(scene_object)
        return True

    def set_object_position(self, scene_object: rt.SceneObject, position) -> bool:
        """Move a scene object to an absolute position."""
        target = np.asarray(position, dtype=float)
        current = scene_object.position.numpy().squeeze()
        if np.allclose(target, current, atol=1e-6):
            return False
        try:
            scene_object.position = mi.Point3f(*target.tolist())
        except (KeyError, RuntimeError):
            # Older scenes do not expose their mesh buffers to sionna
            return self.transform_scene_object(
                scene_object, translation=target - current
            )
        self.scene.scene_geometry_updated()
        self._after_geometry_edit(scene_object)
        return True

    def set_object_orientation(self, scene_object: rt.SceneObject, euler) -> bool:
        """Set a scene object's orientation, in radians."""
        target = np.asarray(euler, dtype=float)
        current = scene_object.orientation.numpy().squeeze()
        if np.allclose(target, current, atol=1e-6):
            return False
        try:
            scene_object.orientation = mi.Point3f(*target.tolist())
        except (KeyError, RuntimeError):
            increment = np.array(
                rotation_matrix(mi.Point3f(*target.tolist())).numpy()
            ).squeeze() @ np.array(
                rotation_matrix(mi.Point3f(*current.tolist())).numpy()
            ).squeeze().T
            return self.transform_scene_object(
                scene_object, rotation_increment=increment
            )
        self.scene.scene_geometry_updated()
        self._after_geometry_edit(scene_object)
        return True

    def update_attached_object(self, device: rt.RadioDevice) -> bool:
        """
        Move the scene object attached to this device (if any) so it stays
        under the device. Updates the physics scene, the rasterized view,
        and marks the ray-traced view for a re-render.
        """
        attachment = self.attachments.get(device.name)
        if attachment is None:
            return False
        scene_object = self.scene.get(attachment["object"])
        if scene_object is None:
            return False

        new_position = device.position.numpy().squeeze() - attachment["offset"]
        delta = new_position - scene_object.position.numpy().squeeze()

        # Rotation of the object follows the device's rotation since attaching
        device_rotation = np.array(
            rotation_matrix(device.orientation).numpy()
        ).squeeze()
        target_rotation = device_rotation @ attachment["device_rotation_0"].T
        # Rotation still to apply, on top of what the object already has
        increment = target_rotation @ attachment["applied_rotation"].T

        needs_translation = not np.allclose(delta, 0.0, atol=1e-6)
        needs_rotation = not np.allclose(increment, np.eye(3), atol=1e-6)
        if not needs_translation and not needs_rotation:
            return False

        # Move the object's vertices in the wireless scene. We go through a
        # fresh traversal because the scene parameters cached by sionna do not
        # expose the mesh buffers for all scenes.
        mesh = scene_object.mi_mesh
        if self._full_scene_params is None:
            self._full_scene_params = mi.traverse(self.scene.mi_scene)
        key = mesh.id() + ".vertex_positions"
        if key not in self._full_scene_params:
            logging.warning("Cannot move scene object %s: no %s", mesh.id(), key)
            return False

        vertices = dr.unravel(mi.Point3f, self._full_scene_params[key])
        if needs_rotation:
            # Rotate around the object's own center, then translate
            center = mi.Point3f(*scene_object.position.numpy().squeeze().tolist())
            vertices = (
                mi.Matrix3f(increment.tolist()) @ (vertices - center)
            ) + center
            attachment["applied_rotation"] = target_rotation
        if needs_translation:
            vertices = vertices + mi.Point3f(*delta.tolist())
        self._full_scene_params[key] = dr.ravel(vertices)
        self._full_scene_params.update()
        # Invalidate sionna's solver caches
        self.scene.scene_geometry_updated()

        # Update the rasterized (polyscope) mesh from the moved vertices
        if ps.has_surface_mesh(mesh.id()):
            ps.get_surface_mesh(mesh.id()).update_vertex_positions(
                mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            )

        self._update_ray_traced_geometry(mesh)
        self.reset_accumulation_requested = True
        return True

    def world_to_screen(self, point) -> tuple[float, float] | None:
        """Project a world point to pixel coordinates, or None if behind."""
        width, height = ps.get_window_size()
        view = ps.get_camera_view_matrix()
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        tan_half_fov = np.tan(np.radians(fov_vertical_deg) / 2.0)
        aspect = width / max(height, 1)

        in_camera = view @ np.append(np.asarray(point, dtype=float), 1.0)
        if in_camera[2] >= -1e-6:
            return None
        u = (in_camera[0] / -in_camera[2]) / (tan_half_fov * aspect)
        v = (in_camera[1] / -in_camera[2]) / tan_half_fov
        return ((u + 1.0) * 0.5 * width, (1.0 - v) * 0.5 * height)

    def device_near_screen_position(self, screen_coords, radius_px: float = 20.0):
        """
        Closest radio device whose marker is within `radius_px` of the given
        screen position. Devices are small and often sit against geometry, so
        clicking near a marker should select it even when something else is
        nominally in front.
        """
        radius = radius_px * self.ui_scale
        best_distance = radius
        best: tuple[rt.RadioDevice, SelectionType] | None = None
        for collection, selection_type in (
            (self.scene._transmitters, SelectionType.Transmitter),
            (self.scene._receivers, SelectionType.Receiver),
        ):
            for device in collection.values():
                projected = self.world_to_screen(device.position.numpy().squeeze())
                if projected is None:
                    continue
                distance = float(
                    np.hypot(
                        projected[0] - screen_coords[0], projected[1] - screen_coords[1]
                    )
                )
                if distance < best_distance:
                    best_distance = distance
                    best = (device, selection_type)
        return best

    def camera_ray(self, screen_coords) -> mi.Ray3f:
        """Ray from the camera through the given screen position."""
        width, height = ps.get_window_size()
        camera_to_world = np.linalg.inv(ps.get_camera_view_matrix())
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        tan_half_fov = np.tan(np.radians(fov_vertical_deg) / 2.0)
        aspect = width / max(height, 1)

        # Normalized device coordinates, with the camera looking down -z
        u = 2.0 * screen_coords[0] / max(width, 1) - 1.0
        v = 1.0 - 2.0 * screen_coords[1] / max(height, 1)
        direction = (
            u * tan_half_fov * aspect * camera_to_world[:3, 0]
            + v * tan_half_fov * camera_to_world[:3, 1]
            - camera_to_world[:3, 2]
        )
        direction /= np.linalg.norm(direction)
        return mi.Ray3f(
            mi.Point3f(*camera_to_world[:3, 3].tolist()),
            mi.Vector3f(*direction.tolist()),
        )

    def resolve_scene_object_at(
        self, screen_coords, pick_result: ps.PickResult | None = None
    ) -> tuple[str | None, np.ndarray | None]:
        """
        Find the scene object under the given screen position, by tracing a ray
        into the scene itself. This works in every rendering mode, including the
        ray-traced view where the rasterized meshes are hidden and so cannot be
        picked by Polyscope.
        """
        si = self.scene.mi_scene.ray_intersect(self.camera_ray(screen_coords))
        if not bool(si.is_valid().numpy().item()):
            return None, None

        point = si.p.numpy().squeeze()
        for name, scene_object in self.scene.objects.items():
            if bool(dr.all(si.shape == mi.ShapePtr(scene_object.mi_mesh))):
                return name, point
        return None, None

    def asset_drop_position(self) -> np.ndarray:
        """
        Where a newly placed asset should land: the point the camera looks at,
        falling back to the center of the scene's ground.
        """
        point = get_point_from_camera_center_ray(
            self.scene, np.linalg.inv(ps.get_camera_view_matrix())
        )
        if point is not None and np.all(np.isfinite(point)):
            return np.asarray(point, dtype=float)
        bbox = self.scene.mi_scene.bbox()
        center = np.array(bbox.center())
        return np.array([center[0], center[1], float(bbox.min.z)])

    def add_asset(self, spec: AssetSpec, position=None) -> rt.SceneObject | None:
        """
        Place a library asset in the scene. It becomes a regular scene object:
        it blocks and reflects signals, can be attached to a radio device, and
        shows up in the object lists.
        """
        if position is None:
            position = self.asset_drop_position()

        index = 1
        while f"{spec.key}-{index}" in self.scene.objects:
            index += 1
        name = f"{spec.key}-{index}"

        try:
            scene_object = rt.SceneObject(
                mi_mesh=build_asset_mesh(spec, name, position),
                name=name,
                radio_material=build_asset_material(spec, self.scene),
            )
            self.scene.edit(add=scene_object)
        except Exception as e:
            logging.error("Could not add asset %s: %s", spec.label, e)
            self._set_export_note(f"Could not add {spec.label}: {e}")
            return None

        self.placed_assets.append(name)
        # Materials are shared, so the viewer colour is kept per asset
        self.asset_colors[name] = spec.color
        self.on_scene_geometry_changed()
        self._set_export_note(f"Placed {spec.label} ({name})")
        return scene_object

    def remove_asset(self, name: str) -> None:
        """Remove a previously placed asset from the scene."""
        # Detach any device that was following it
        for device_name, attachment in list(self.attachments.items()):
            if attachment["object"] == name:
                del self.attachments[device_name]
        try:
            mesh_id = self.scene.get(name).mi_mesh.id()
            self.scene.edit(remove=name)
        except Exception as e:
            logging.error("Could not remove asset %s: %s", name, e)
            return
        if name in self.placed_assets:
            self.placed_assets.remove(name)
        self.asset_colors.pop(name, None)
        if ps.has_surface_mesh(mesh_id):
            ps.get_surface_mesh(mesh_id).remove()
        self.on_scene_geometry_changed()

    def on_scene_materials_changed(self) -> None:
        """Refresh what depends on radio materials after one was changed."""
        # The ray-traced view builds its own scene from the materials
        self.render_cache = None
        self.reset_accumulation_requested = True
        self.reset_radio_map()
        if self.cfg.paths.auto_update:
            self.update_paths(clear_first=True, show=True)

    def on_scene_geometry_changed(self) -> None:
        """
        Refresh everything that caches the scene's geometry after objects were
        added or removed.
        """
        add_scene_to_polyscope(self.scene, self.ps_groups)
        # Meshes are coloured from their material there, so restore the
        # per-asset colours afterwards
        for asset_name, color in self.asset_colors.items():
            scene_object = self.scene.get(asset_name)
            if scene_object is None:
                continue
            mesh_id = scene_object.mi_mesh.id()
            if ps.has_surface_mesh(mesh_id):
                ps.get_surface_mesh(mesh_id).set_color(color)
        shapes = self.scene.mi_scene.shapes()
        self.scene_stats = (len(shapes), sum(sh.face_count() for sh in shapes))
        # The ray-traced view builds its own copy of the scene, and our
        # traversal of the wireless scene is now stale.
        self.render_cache = None
        self._full_scene_params = None
        self.reset_accumulation_requested = True
        self.reset_radio_map()
        if self.cfg.paths.auto_update:
            self.update_paths(clear_first=True, show=True)

    def assets_gui(self) -> None:
        """Asset library: place ready-made objects into the scene."""
        psim.TextDisabled("Placed where the camera is looking.")
        for i, spec in enumerate(ASSET_LIBRARY):
            if i % 3 != 0:
                psim.SameLine()
            if psim.Button(f"{spec.label}##asset_{spec.key}"):
                self.add_asset(spec)
            if psim.IsItemHovered():
                size = spec.size
                psim.SetTooltip(
                    f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m\n{spec.note}"
                )

        if not self.placed_assets:
            return
        psim.Spacing()
        for name in list(self.placed_assets):
            psim.PushID(name)
            psim.AlignTextToFramePadding()
            psim.Text(name)
            psim.SameLine()
            psim.SetCursorPosX(
                psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - 24 * self.ui_scale
            )
            if psim.SmallButton("x"):
                self.remove_asset(name)
            psim.PopID()

    def start_attach_pick(self, device_name: str) -> None:
        """
        Enter "click an object to attach to" mode. Objects are only visible and
        pickable in the rasterized view, so switch to it for the duration and
        restore the previous rendering mode afterwards.
        """
        self.attach_pick_pending = device_name
        self._attach_previous_rendering_mode = self.cfg.rendering.mode
        if self.cfg.rendering.mode != RenderingMode.RASTERIZATION:
            self.set_rendering_mode(RenderingMode.RASTERIZATION)

    def end_attach_pick(self) -> None:
        """Leave object-picking mode and restore the previous rendering mode."""
        self.attach_pick_pending = None
        self.clear_attach_highlight()
        previous_mode = self._attach_previous_rendering_mode
        self._attach_previous_rendering_mode = None
        if previous_mode is not None and previous_mode != self.cfg.rendering.mode:
            self.set_rendering_mode(previous_mode)

    def clear_attach_highlight(self) -> None:
        self.attach_hover_name = None
        if ps.has_curve_network(ATTACH_HIGHLIGHT_NAME):
            ps.get_curve_network(ATTACH_HIGHLIGHT_NAME).remove()
        # Restore the color of the previously tinted mesh
        if self._attach_tinted is not None:
            mesh_id, color = self._attach_tinted
            self._attach_tinted = None
            if ps.has_surface_mesh(mesh_id):
                ps.get_surface_mesh(mesh_id).set_color(color)

    def _tint_hovered_mesh(self, mesh_id: str) -> None:
        """
        Tint the hovered mesh with the accent color (restoring the previously
        tinted one). This reads more clearly than the outline alone, which is
        depth-tested and can hide behind other geometry.
        """
        if self._attach_tinted is not None and self._attach_tinted[0] == mesh_id:
            return
        if self._attach_tinted is not None:
            previous_id, previous_color = self._attach_tinted
            if ps.has_surface_mesh(previous_id):
                ps.get_surface_mesh(previous_id).set_color(previous_color)
            self._attach_tinted = None
        if ps.has_surface_mesh(mesh_id):
            struct = ps.get_surface_mesh(mesh_id)
            self._attach_tinted = (mesh_id, struct.get_color())
            struct.set_color(ACCENT_BRIGHT)

    def update_attach_highlight(self) -> None:
        """
        While picking an object to attach to, outline the object under the
        cursor and label it, so it is clear what a click would select.
        """
        if self.attach_pick_pending is None:
            self.clear_attach_highlight()
            return

        imgui_io = psim.GetIO()
        if imgui_io.WantCaptureMouse:
            # Cursor is over the GUI, not the scene
            self.clear_attach_highlight()
            return

        object_name, _ = self.resolve_scene_object_at(imgui_io.MousePos)
        if object_name is None:
            self.clear_attach_highlight()
            return
        self.attach_hover_name = object_name

        scene_object = self.scene.get(object_name)
        self._tint_hovered_mesh(scene_object.mi_mesh.id())

        # Outline the object's bounding box (the only cue available in
        # ray-traced mode, where the rasterized meshes are hidden)
        bbox = scene_object.mi_mesh.bbox()
        low, high = np.array(bbox.min), np.array(bbox.max)
        corners = np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        # Corner i encodes (x, y, z) choices in its bits: connect the pairs
        # that differ in exactly one axis.
        edges = np.array(
            [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i & bit == 0]
        )
        struct = ps.register_curve_network(
            ATTACH_HIGHLIGHT_NAME, corners, edges, color=ACCENT_BRIGHT, enabled=True
        )
        struct.set_radius(
            max(0.0012 * scene_scale(self.scene), 0.15), relative=False
        )
        struct.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)

        # Label next to the cursor
        psim.GetForegroundDrawList().AddText(
            (imgui_io.MousePos[0] + 18 * self.ui_scale, imgui_io.MousePos[1]),
            im_col32(*ACCENT_BRIGHT),
            object_name,
        )

    def radio_device_list_gui(self) -> None:
        """
        List of all radio devices in the scene: click a row to select the
        device (showing its gizmo), click 'x' to remove it.
        """
        if not self.scene._transmitters and not self.scene._receivers:
            psim.TextDisabled("No radio devices in the scene yet.")
            return

        to_remove: list[tuple[rt.RadioDevice, bool]] = []
        for is_transmitter, devices in (
            (True, self.scene._transmitters),
            (False, self.scene._receivers),
        ):
            for name, rd in devices.items():
                psim.PushID(name)
                is_selected = self.selected_object is rd
                swatch = psim.GetFontSize() * 0.9
                psim.ColorButton(
                    "##swatch",
                    (*rd.color, 1.0),
                    psim.ImGuiColorEditFlags_NoTooltip,
                    (swatch, swatch),
                )
                psim.SameLine()
                row_width = (
                    psim.GetContentRegionAvail()[0] - 30 * self.ui_scale
                )
                clicked = psim.Selectable(name, is_selected, 0, (row_width, 0))
                if psim.IsItemHovered():
                    position = rd.position.numpy().squeeze()
                    psim.SetTooltip(
                        f"{'Transmitter' if is_transmitter else 'Receiver'} at "
                        f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}).\n"
                        "Click to select and show its move gizmo."
                    )
                if clicked:
                    if is_selected:
                        self.clear_selection()
                    else:
                        self.selected_object = rd
                        self.selected_type = (
                            SelectionType.Transmitter
                            if is_transmitter
                            else SelectionType.Receiver
                        )
                psim.SameLine()
                if psim.SmallButton("x"):
                    to_remove.append((rd, is_transmitter))
                psim.PopID()

        for rd, is_transmitter in to_remove:
            if self.selected_object is rd:
                self.clear_selection()
            self.remove_object(
                rd,
                SelectionType.Transmitter if is_transmitter else SelectionType.Receiver,
            )
            set_or_update_radio_devices_polyscope(
                self.scene.transmitters if is_transmitter else self.scene.receivers,
                is_transmitter,
                self,
            )
            if is_transmitter:
                self.reset_radio_map()
            if self.cfg.paths.auto_update:
                self.update_paths(clear_first=True, show=True)

    def clear_radio_devices(self) -> None:
        for name in self.scene._transmitters.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]
        for name in self.scene._receivers.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]

        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()
        if self.selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
            self.clear_selection()

        if self.cfg.radio_map.auto_update:
            self.clear_radio_map()
        if self.cfg.paths.auto_update:
            self.clear_paths()

    def reset_radio_map(self):
        """
        Attempts to reset the radio map to zero, as well as the accumulation counter.
        However, if the number of transmitters has changed, the radio map will be
        completely removed (and re-computed if auto-updates are enabled).
        """
        self.rm_accumulated_samples = 0
        if self.radio_map is None:
            return

        if len(self.scene._transmitters) != self.radio_map.num_tx:
            self.clear_radio_map()
            if self.cfg.radio_map.auto_update:
                self.set_radio_map(self.compute_radio_map(), show=True)
            return

        self.radio_map._pathgain_map *= 0.0

    def set_rm_probe(self, world_position: np.ndarray) -> None:
        """
        Place the radio map probe at the clicked world position: the value of
        the underlying cell is shown in the Radio map section.
        """
        if self.radio_map is None or not isinstance(self.radio_map, rt.PlanarRadioMap):
            return
        to_local = np.linalg.inv(self.radio_map.to_world.matrix.numpy().squeeze())
        local = to_local @ np.append(world_position, 1.0)
        height, width = self.radio_map.path_gain.shape[1:]
        j = int(round((local[0] + 1.0) / 2.0 * (width - 1)))
        i = int(round((local[1] + 1.0) / 2.0 * (height - 1)))
        if not (0 <= i < height and 0 <= j < width):
            return
        self.rm_probe = (i, j, np.asarray(world_position))

        marker = ps.register_point_cloud(
            "RM probe",
            np.asarray(world_position)[None, :],
            color=ACCENT_BRIGHT,
            enabled=True,
        )
        marker.set_radius(
            max(0.0015 * scene_scale(self.scene), 1.2), relative=False
        )
        marker.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME, True)

    def rm_probe_value_db(self) -> float | None:
        """Current path gain in dB at the probe's cell (max over transmitters)."""
        if self.rm_probe is None or self.radio_map is None:
            return None
        i, j, _ = self.rm_probe
        path_gain = self.radio_map.path_gain
        n_tx, height, width = path_gain.shape
        if not (0 <= i < height and 0 <= j < width):
            return None
        # Read the probed cell for each transmitter from the flat array
        indices = mi.UInt32([t * height * width + i * width + j for t in range(n_tx)])
        value = float(dr.max(dr.gather(mi.Float, path_gain.array, indices))[0])
        if value <= 0.0:
            return None
        return 10.0 * np.log10(value)

    def clear_rm_probe(self) -> None:
        self.rm_probe = None
        if ps.has_point_cloud("RM probe"):
            ps.get_point_cloud("RM probe").remove()

    def clear_radio_map(self):
        self.radio_map = None
        self.rm_accumulated_samples = 0
        self.rm_colorbar = None
        self.rm_colorbar_texture_id = None
        self.clear_rm_probe()
        if ps.has_surface_mesh("radio_map"):
            ps.get_surface_mesh("radio_map").remove()

    def clear_paths(self):
        self.paths = None
        self.paths_taps = None
        self.paths_cir = None
        if ps.has_curve_network("paths"):
            ps.get_curve_network("paths").remove()

    # ------------------------

    def set_slice_plane_active(self, active: bool):
        if self.slice_plane is None:
            return
        self.slice_plane.set_active(active)

        if active and self.cfg.rendering.mode == RenderingMode.RAY_TRACING:
            # Switch to rasterization mode when activating the slice plane.
            self.set_rendering_mode(RenderingMode.RASTERIZATION)

    # ------------------------

    def process_inputs(self):
        imgui_io = psim.GetIO()
        if imgui_io.WantCaptureKeyboard:
            # If the user is e.g. typing in a text field, don't process
            # keyboard or mouse inputs.
            return

        allow_click = not imgui_io.WantCaptureMouse
        has_left_click = allow_click and psim.IsMouseClicked(psim.ImGuiMouseButton_Left)
        has_mouse_drag = psim.IsMouseDragging(
            psim.ImGuiMouseButton_Left
        ) or psim.IsMouseDragging(psim.ImGuiMouseButton_Right)
        has_left_release = allow_click and psim.IsMouseReleased(
            psim.ImGuiMouseButton_Left
        )
        has_right_click = allow_click and psim.IsMouseClicked(
            psim.ImGuiMouseButton_Right
        )
        has_right_release = allow_click and psim.IsMouseReleased(
            psim.ImGuiMouseButton_Right
        )
        has_active_item = psim.IsAnyItemActive()

        # Plain right click (without dragging, which pans the camera) opens the
        # context menu for whatever is under the cursor.
        if (
            has_right_release
            and not has_mouse_drag
            and not self.was_mouse_dragging
            and not (imgui_io.KeyCtrl or imgui_io.KeyShift or imgui_io.KeyAlt)
        ):
            self.request_context_menu(imgui_io.MousePos)

        # TODO: +/- to zoom in/out
        # TODO: keyboard shortcuts to move around (WASD + QE)
        # TODO: keyboard shortcuts to rotate the envmap?

        # K/L or (Ctrl + left/right click): add transmitter/receiver
        has_k = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("K")))
        has_l = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("L")))
        if (
            has_k
            or has_l
            or (
                imgui_io.KeyCtrl
                and (has_left_click or has_right_click)
                and not imgui_io.KeyShift
            )
        ):
            is_transmitter = imgui_io.MouseClicked[0] or has_k

            origin = ps.get_camera_view_matrix()[:3, 3]
            rd_position = ps.screen_coords_to_world_position(imgui_io.MousePos)

            if np.all(np.isfinite(rd_position)):
                normal = get_normal_for_path(self.scene, origin, rd_position)
                if normal is None:
                    normal = np.array([0, 0, 1])
                # Make sure that normal is oriented towards the camera
                if np.dot(normal, rd_position - origin) < 0:
                    normal = -normal
                rd_position += 1.5 * normal

                self.add_radio_device(rd_position, is_transmitter)

        # Plain left click: object selection
        if has_left_release and not (
            imgui_io.KeyCtrl
            or imgui_io.KeyShift
            or imgui_io.KeyAlt
            or has_mouse_drag
            or self.was_mouse_dragging
        ):
            self.process_pick_result(ps.pick(screen_coords=imgui_io.MousePos))

        # Shift + R: reload code
        if imgui_io.KeyShift and psim.IsKeyPressed(
            psim.ImGuiKey(ps.get_key_code("R")), repeat=False
        ):
            self.code_reload_requested = True

        # R: reset camera to initial position
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("R")), repeat=False):
            self.move_camera_home()
        # F: fit scene in camera viewport
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("F")), repeat=False):
            self.fit_camera_to_scene()

        # C: go to next rendering mode.
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("C")), repeat=False):
            self.set_rendering_mode(
                RenderingMode((self.cfg.rendering.mode.value + 1) % len(RenderingMode))
            )

        # H / ?: show help window
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("H")), repeat=False) or (
            imgui_io.KeyShift
            and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("/")), repeat=False)
        ):
            self.cfg.show_help_window = not self.cfg.show_help_window

        # M: toggle radio map computation and display
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("M")), repeat=False):
            rm_visible, rm_struct = self.has_visible_radio_map()

            if rm_visible:
                # Hide radio map (but don't delete it, we may want to show it later)
                self.cfg.radio_map.auto_update = False
                rm_struct.set_enabled(False)
            else:
                self.cfg.radio_map.auto_update = True
                if self.radio_map is not None:
                    # Show the existing radio map
                    rm_struct.set_enabled(True)
                else:
                    # Compute and show the radio map
                    self.set_radio_map(self.compute_radio_map(), show=True)

        if self.slice_plane is not None:
            # S: toggle slice plane
            if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("S")), repeat=False):
                self.set_slice_plane_active(not self.slice_plane.get_active())

            # Alt + left click drag: move slice plane along its normal.
            # TODO: avoid using middle click, since it's not easy to do on trackpads.
            if self.slice_plane.get_active() and imgui_io.KeyAlt and has_mouse_drag:
                ps.set_do_default_mouse_interaction(False)
                center = self.slice_plane.get_center()
                normal = self.slice_plane.get_normal()
                self.slice_plane.set_pose(
                    center + 0.3 * self.ui_scale * imgui_io.MouseDelta[1] * normal,
                    normal,
                )
            else:
                ps.set_do_default_mouse_interaction(True)

        # Tab: toggle show GUI (ours)
        if not has_active_item and psim.IsKeyPressed(psim.ImGuiKey_Tab, repeat=False):
            self.cfg.gui_mode = GuiMode((self.cfg.gui_mode.value + 1) % len(GuiMode))

        # Esc: cancel attach-picking, close help window, or de-select
        if psim.IsKeyPressed(psim.ImGuiKey_Escape, repeat=False):
            if self.attach_pick_pending is not None:
                self.end_attach_pick()
            elif self.cfg.show_help_window:
                self.cfg.show_help_window = False
            elif self.selected_object is not None:
                self.clear_selection()

        # Ctrl + Q: exit
        if imgui_io.KeyCtrl and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("Q"))):
            ps.unshow()

        self.was_mouse_dragging = has_mouse_drag

    def process_pick_result(self, pick_result: ps.PickResult) -> bool:
        # One-shot "pick an object to attach to" mode
        if self.attach_pick_pending is not None:
            device_name = self.attach_pick_pending
            self.end_attach_pick()
            device = self.scene.get(device_name)
            if device is not None:
                object_name, snap_position = self.resolve_scene_object_at(
                    pick_result.screen_coords, pick_result=pick_result
                )
                if object_name is not None:
                    self.attach_device_to_object(
                        device, object_name, snap_position=snap_position
                    )
                    self.update_attached_object(device)
                    set_or_update_radio_devices_polyscope(
                        self.scene.transmitters
                        if isinstance(device, rt.Transmitter)
                        else self.scene.receivers,
                        isinstance(device, rt.Transmitter),
                        self,
                    )
                    if isinstance(device, rt.Transmitter):
                        self.reset_radio_map()
                    if self.cfg.paths.auto_update:
                        self.update_paths(clear_first=True, show=True)
            return True

        if pick_result.is_hit and pick_result.structure_name == "radio_map":
            self.set_rm_probe(pick_result.position)
            return True

        # A radio device: the point clouds carry the device index. Match the
        # structure name exactly, since scene meshes also report an index.
        devices = {
            "Transmitters": (self.scene._transmitters, SelectionType.Transmitter),
            "Receivers": (self.scene._receivers, SelectionType.Receiver),
        }
        if pick_result.is_hit and pick_result.structure_name in devices:
            collection, selection_type = devices[pick_result.structure_name]
            index = pick_result.structure_data.get("index")
            values = list(collection.values())
            if index is not None and 0 <= index < len(values):
                self.select_radio_device(values[index], selection_type)
                return True

        # A device whose marker is under the cursor but which Polyscope did not
        # report, because it sits against or inside geometry
        nearby = self.device_near_screen_position(pick_result.screen_coords)
        if nearby is not None:
            self.select_radio_device(*nearby)
            return True

        # Otherwise a scene object, resolved by tracing into the scene so that
        # it works in the ray-traced view too
        object_name, _ = self.resolve_scene_object_at(
            pick_result.screen_coords, pick_result=pick_result
        )
        if object_name is not None and object_name not in self.background_objects:
            self.select_scene_object(object_name)
            return True

        # The ground plane counts as empty space: selecting it would put the
        # gizmo at the middle of the whole scene, which is never what a click
        # on the ground means.
        self.clear_selection()
        return False

    def request_context_menu(self, screen_coords) -> None:
        """Remember what was under the cursor, and open the context menu."""
        world = ps.screen_coords_to_world_position(screen_coords)
        self.context_menu_position = (
            np.asarray(world, dtype=float) if np.all(np.isfinite(world)) else None
        )
        object_name, _ = self.resolve_scene_object_at(screen_coords)
        self.context_menu_object = object_name
        self.context_menu_requested = True

    def viewport_context_menu(self) -> None:
        """Right-click menu: act on the point and object under the cursor."""
        if self.context_menu_requested:
            psim.OpenPopup("##viewport_context")
            self.context_menu_requested = False

        if not psim.BeginPopup("##viewport_context"):
            return

        position = self.context_menu_position
        has_position = position is not None
        if has_position:
            psim.TextDisabled(
                f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
            )
        else:
            psim.TextDisabled("Empty space")
        psim.Separator()

        psim.BeginDisabled(not has_position)
        if psim.MenuItem("Add transmitter here"):
            self.add_radio_device(position + [0.0, 0.0, 1.5], is_transmitter=True)
        if psim.MenuItem("Add receiver here"):
            self.add_radio_device(position + [0.0, 0.0, 1.5], is_transmitter=False)
        if psim.BeginMenu("Add asset here"):
            for spec in ASSET_LIBRARY:
                if psim.MenuItem(spec.label):
                    self.add_asset(spec, position=position)
            psim.EndMenu()
        psim.EndDisabled()

        if self.context_menu_object is not None:
            psim.Separator()
            name = self.context_menu_object
            if psim.MenuItem(f"Select '{name}'"):
                self.select_scene_object(name)
            if name in self.placed_assets and psim.MenuItem(f"Remove '{name}'"):
                self.remove_asset(name)

        if self.radio_map is not None and has_position:
            psim.Separator()
            if psim.MenuItem("Probe radio map here"):
                self.set_rm_probe(position)

        psim.Separator()
        if psim.MenuItem("Fit view"):
            self.fit_camera_to_scene()
        if psim.MenuItem("Top view"):
            self.move_camera_top()
        psim.EndPopup()

    def set_look_at_target(self, device: rt.RadioDevice, target_name: str | None) -> None:
        """
        Keep a device pointed at another device or a scene object. Sionna's
        look_at only sets the orientation once, so the target is remembered and
        re-applied whenever either of them moves.
        """
        if target_name is None:
            self.look_at_targets.pop(device.name, None)
            self._look_at_state.pop(device.name, None)
            return
        self.look_at_targets[device.name] = target_name
        self._look_at_state.pop(device.name, None)
        self.apply_look_at_targets(force=True)

    def look_at_target_position(self, target_name: str) -> np.ndarray | None:
        """Where a look-at target currently is."""
        target = self.scene.get(target_name)
        if target is None:
            return None
        return target.position.numpy().squeeze()

    def apply_look_at_targets(self, force: bool = False) -> None:
        """
        Re-point every device that is tracking something. Nothing is touched
        unless the device or its target has actually moved, so a locked device
        costs nothing while the scene is still.
        """
        if not self.look_at_targets:
            return

        tx_changed = False
        rx_changed = False
        for device_name, target_name in list(self.look_at_targets.items()):
            device = self.scene.get(device_name)
            target_position = self.look_at_target_position(target_name)
            if device is None or target_position is None:
                self.look_at_targets.pop(device_name, None)
                continue

            device_position = device.position.numpy().squeeze()
            state = (tuple(device_position), tuple(target_position))
            if not force and self._look_at_state.get(device_name) == state:
                continue
            self._look_at_state[device_name] = state

            if np.allclose(device_position, target_position, atol=1e-6):
                continue
            device.look_at(mi.Point3f(*[float(v) for v in target_position]))
            dr.make_opaque(device.orientation)
            if isinstance(device, rt.Transmitter):
                tx_changed = True
            else:
                rx_changed = True

        propagate_device_updates(self, tx_changed, rx_changed)

    def select_radio_device(self, device, selection_type) -> None:
        """Make a radio device the active object."""
        self.clear_selection()
        self.selected_object = device
        self.selected_type = selection_type
        self.properties_tab = PROPERTIES_TABS.index("Object")

    def select_scene_object(self, object_name: str) -> None:
        """Make a scene object (a building, a placed asset, ...) the active one."""
        scene_object = self.scene.get(object_name)
        if scene_object is None:
            return
        self.clear_selection()
        self.selected_object = scene_object
        self.selected_type = SelectionType.Mesh
        self.properties_tab = PROPERTIES_TABS.index("Object")

    def clear_selection(self):
        self.selected_object = None
        self.selected_type = None
        self.object_gizmo_previous = None
        if ps.has_point_cloud("Gizmo"):
            ps.get_point_cloud("Gizmo").remove()
        if ps.has_curve_network("Trajectory"):
            ps.get_curve_network("Trajectory").remove()
        if ps.has_point_cloud("Trajectory waypoints"):
            ps.get_point_cloud("Trajectory waypoints").remove()
        remove_antenna_pattern_structure()
        self.antenna_pattern_cache_key = None

    def _export_directory(self) -> str:
        pictures = os.path.join(os.path.expanduser("~"), "Pictures")
        return pictures if os.path.isdir(pictures) else os.getcwd()

    def _set_export_note(self, note: str) -> None:
        self.last_export_note = (note, time.time())

    def save_screenshot(self) -> None:
        """Save a PNG of the current view (including the UI) to disk."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._export_directory(), f"sionna_rt_{stamp}.png")
        ps.screenshot(filename=path, transparent_bg=False, include_UI=True)
        self._set_export_note(f"Saved {path}")

    def export_radio_map(self) -> None:
        """Save the current radio map (path gain, linear) as a .npy file."""
        if self.radio_map is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._export_directory(), f"radio_map_{stamp}.npy")
        np.save(path, self.radio_map._pathgain_map.numpy())
        self._set_export_note(f"Saved {path}")

    def set_carrier_frequency(self, frequency_hz: float) -> None:
        """
        Set the scene's carrier frequency and refresh all radio results,
        since material properties and propagation depend on it.
        """
        self.scene.frequency = frequency_hz
        self.reset_radio_map()
        if self.cfg.paths.auto_update:
            self.update_paths(clear_first=True, show=True)

    def frequency_gui(self) -> None:
        """
        Carrier frequency control: free input in GHz plus common-band presets.
        """
        frequency_ghz = float(self.scene.frequency[0]) / 1e9

        property_row("Carrier frequency [GHz]", self.ui_scale)
        changed, new_frequency_ghz = psim.InputFloat(
            "##carrier_frequency",
            frequency_ghz,
            format="%.3f",
            flags=psim.ImGuiInputTextFlags_EnterReturnsTrue,
        )
        end_property_row()
        if psim.IsItemHovered():
            psim.SetTooltip(
                "Press Enter to apply. Radio materials and propagation\n"
                "results depend on the carrier frequency."
            )
        if changed and 0.001 <= new_frequency_ghz <= 1000:
            self.set_carrier_frequency(new_frequency_ghz * 1e9)

        # Common band presets
        for label, preset_ghz in (
            ("2.4", 2.4),
            ("3.5", 3.5),
            ("5.9", 5.9),
            ("28", 28.0),
            ("60", 60.0),
        ):
            is_current = abs(frequency_ghz - preset_ghz) < 1e-6
            if is_current:
                psim.PushStyleColor(
                    psim.ImGuiCol_Button, (0.30, 0.46, 0.04, 1.0)
                )
            if psim.Button(f"{label}##freq_preset"):
                self.set_carrier_frequency(preset_ghz * 1e9)
            if is_current:
                psim.PopStyleColor()
            psim.SameLine()
        wavelength_m = 299792458.0 / (frequency_ghz * 1e9)
        if wavelength_m >= 0.1:
            wavelength_str = f"{100 * wavelength_m:.1f} cm"
        else:
            wavelength_str = f"{1000 * wavelength_m:.1f} mm"
        psim.TextDisabled(f"wavelength {wavelength_str}")

    def header_gui(self):
        """
        Branded header of the main panel: app title, accent underline, and a
        live status row (FPS health, device counts, current scene).
        """
        draw_list = psim.GetWindowDrawList()

        # Title line, with the snapshot and help buttons on the right
        psim.TextColored((*ACCENT_BRIGHT, 1.0), "S I O N N A   R T")
        psim.SameLine()
        bw = 26 * self.ui_scale
        buttons_width = 118 * self.ui_scale
        psim.SetCursorPosX(
            psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - buttons_width
        )
        if psim.Button("Save view"):
            self.save_screenshot()
        if psim.IsItemHovered():
            psim.SetTooltip("Save a PNG screenshot of the current view")
        psim.SameLine()
        if psim.Button("?", size=(bw, 0)):
            self.cfg.show_help_window = not self.cfg.show_help_window
        if psim.IsItemHovered():
            psim.SetTooltip("Controls & shortcuts (H)")

        psim.TextDisabled("Ray-traced radio propagation")
        psim.SameLine()
        back_label = "Docked panels"
        style = psim.GetStyle()
        back_width = psim.CalcTextSize(back_label)[0] + 2 * style.FramePadding[0]
        psim.SetCursorPosX(
            max(
                psim.GetCursorPosX(),
                psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - back_width,
            )
        )
        if psim.Button(f"{back_label}##header"):
            self.set_docked_layout(True)
        if psim.IsItemHovered():
            psim.SetTooltip("Switch back to the docked workspace layout")

        # Accent underline
        x, y = psim.GetCursorScreenPos()
        width = psim.GetContentRegionAvail()[0]
        draw_list.AddRectFilled(
            (x, y + 2 * self.ui_scale),
            (x + width, y + 4 * self.ui_scale),
            im_col32(*ACCENT),
            1.0,
        )
        psim.Dummy((0, 8 * self.ui_scale))

        # Status row: FPS health dot, device counts, current scene
        io = psim.GetIO()
        fps = io.Framerate
        if fps >= 30:
            fps_color = ACCENT_BRIGHT
        elif fps >= 15:
            fps_color = (0.95, 0.75, 0.20)
        else:
            fps_color = (0.90, 0.30, 0.25)

        cx, cy = psim.GetCursorScreenPos()
        radius = 3.5 * self.ui_scale
        draw_list.AddCircleFilled(
            (cx + radius, cy + 0.55 * psim.GetFontSize()), radius, im_col32(*fps_color)
        )
        psim.Dummy((2.5 * radius, 0))
        psim.SameLine()
        psim.Text(f"{fps:.0f} FPS")
        if psim.IsItemHovered():
            psim.SetTooltip(f"Frame time: {1000 * io.DeltaTime:.2f} ms")

        psim.SameLine()
        psim.TextDisabled("|")
        psim.SameLine()
        psim.Text(
            f"{len(self.scene._transmitters)} TX, {len(self.scene._receivers)} RX"
        )

        if 0 <= self.current_scene_idx < len(self.known_scene_names):
            psim.SameLine()
            psim.TextDisabled("|")
            psim.SameLine()
            psim.TextDisabled(self.known_scene_names[self.current_scene_idx])

        # Transient confirmation after exports ("Saved /path/to/file")
        note = getattr(self, "last_export_note", None)
        if note is not None and time.time() - note[1] < 8.0:
            psim.TextColored((*ACCENT_BRIGHT, 1.0), note[0])

        psim.Dummy((0, 2 * self.ui_scale))

    def section_scene(self, include_picker: bool = True, include_camera: bool = True) -> None:
        """
        Scene settings. The picker and camera presets are skipped when the top
        bar already carries them, to avoid duplicate controls.
        """
        if psim.CollapsingHeader("Scene", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()

            if include_picker:
                # Quick pick from built-in or recently-loaded scenes.
                psim.Text("Scene selection:")
                changed, combo_i = psim.Combo(
                    "##scene_picker",
                    self.current_scene_idx,
                    self.known_scene_names,
                )
                if changed:
                    self.load_scene_requested = self.known_scene_paths[combo_i]

            n_meshes, n_triangles = getattr(self, "scene_stats", (0, 0))
            if n_triangles >= 1_000_000:
                triangles_str = f"{n_triangles / 1e6:.1f}M"
            elif n_triangles >= 1_000:
                triangles_str = f"{n_triangles / 1e3:.0f}k"
            else:
                triangles_str = str(n_triangles)
            psim.TextDisabled(f"{n_meshes} meshes, {triangles_str} triangles")

            psim.Spacing()

            if include_camera:
                psim.AlignTextToFramePadding()
                psim.Text("Camera:")
                psim.SameLine()
                if psim.Button("Top##camera"):
                    self.move_camera_top()
                psim.SameLine()
                if psim.Button("Fit##camera"):
                    self.fit_camera_to_scene()
                if psim.IsItemHovered():
                    psim.SetTooltip("Fit the whole scene in view (F)")
                psim.SameLine()
                if psim.Button("Home##camera"):
                    self.move_camera_home()
                if psim.IsItemHovered():
                    psim.SetTooltip("Return to the initial view (R)")

                psim.Spacing()
            self.frequency_gui()
            psim.Spacing()

            if psim.TreeNodeEx(
                "Signal and noise##scene", psim.ImGuiTreeNodeFlags_DefaultOpen
            ):
                noise_contents(self)
                psim.TreePop()
            psim.Spacing()

    def section_assets(self) -> None:
        if psim.CollapsingHeader("Assets"):
            psim.Spacing()
            self.assets_gui()
            psim.Spacing()

    def section_devices(self) -> None:
        n_tx = len(self.scene._transmitters)
        n_rx = len(self.scene._receivers)
        if psim.CollapsingHeader(
            f"Radio devices ({n_tx} TX, {n_rx} RX)###radio_devices",
            psim.ImGuiTreeNodeFlags_DefaultOpen,
        ):
            psim.Spacing()
            # TODO: button to place radio devices: at random; or samples from a radio map
            if psim.Button("+ Add transmitter"):
                self.add_radio_device(
                    self.default_new_device_position(is_transmitter=True),
                    is_transmitter=True,
                )
            psim.SameLine()
            if psim.Button("+ Add receiver"):
                self.add_radio_device(
                    self.default_new_device_position(is_transmitter=False),
                    is_transmitter=False,
                )
            psim.TextDisabled(
                f"Tip: {CTRL_OR_CMD} + left / right click in the scene places\n"
                "a transmitter / receiver at the clicked point."
            )
            psim.Spacing()

            self.radio_device_list_gui()
            psim.Spacing()

            antenna_array_gui(self)

            psim.PushStyleColor(psim.ImGuiCol_Button, (0.42, 0.16, 0.14, 1.0))
            psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.55, 0.20, 0.17, 1.0))
            psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.65, 0.24, 0.20, 1.0))
            clicked = psim.Button("Remove all devices")
            psim.PopStyleColor(3)
            if clicked:
                self.clear_radio_devices()

            psim.Spacing()

    def section_radio_map(self) -> None:
        if psim.CollapsingHeader("Radio map", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False
            needs_visual_update = False

            if psim.Button("Compute radio map"):
                self.set_radio_map(self.compute_radio_map(), show=True)

            psim.SameLine()
            psim.BeginDisabled(self.radio_map is None)
            if psim.Button("Remove##radio_map"):
                self.clear_radio_map()
            psim.EndDisabled()

            psim.SameLine()
            psim.BeginDisabled(self.radio_map is None)
            if psim.Button("Export##radio_map"):
                self.export_radio_map()
            if psim.IsItemHovered(psim.ImGuiHoveredFlags_AllowWhenDisabled):
                psim.SetTooltip(
                    "Save the path gain map (linear, one layer per\n"
                    "transmitter) as a NumPy .npy file."
                )
            psim.EndDisabled()

            psim.SameLine()
            _, self.cfg.radio_map.auto_update = psim.Checkbox(
                "Automatic update##rm", self.cfg.radio_map.auto_update
            )

            if self.rm_probe is not None:
                probe_db = self.rm_probe_value_db()
                position = self.rm_probe[2]
                psim.TextColored(
                    (*ACCENT_BRIGHT, 1.0),
                    f"Probe: {probe_db:.1f} dB"
                    if probe_db is not None
                    else "Probe: no coverage",
                )
                psim.SameLine()
                psim.TextDisabled(
                    f"at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
                )
                psim.SameLine()
                if psim.SmallButton("x##clear_probe"):
                    self.clear_rm_probe()
            elif self.radio_map is not None:
                psim.TextDisabled("Tip: click the radio map to probe its value.")

            if self.radio_map is not None:
                refresh_statistics(self)
                stats = self.radio_map_stats
                if stats and stats.get("covered"):
                    psim.TextDisabled(
                        f"Median {stats['rss_dbm_median']:.0f} dBm over "
                        f"{stats['covered']} cells - see the Analysis tab"
                    )

            # -- Radio map computation options
            psim.Spacing()

            property_row("Cell size [m]", self.ui_scale)
            changed, self.cfg.radio_map.cell_size = psim.InputFloat2(
                "##rm_cell_size",
                self.cfg.radio_map.cell_size,
                format="%.2f",
            )
            end_property_row()
            if changed:
                self.cfg.radio_map.cell_size = tuple(
                    min(max(v, 0.01), 100) for v in self.cfg.radio_map.cell_size
                )
            needs_update |= changed

            if psim.TreeNodeEx("Solver##rm_solver"):
                use_threshold = self.cfg.radio_map.stop_threshold_db is not None
                changed, use_threshold = psim.Checkbox(
                    "Stop weak rays##rm_stop", use_threshold
                )
                if changed:
                    self.cfg.radio_map.stop_threshold_db = -60.0 if use_threshold else None
                    needs_update = True
                if psim.IsItemHovered():
                    psim.SetTooltip(
                        "Abandon a ray once its contribution has fallen this far.\n"
                        "Cheaper, at the cost of the very weakest contributions."
                    )
                if self.cfg.radio_map.stop_threshold_db is not None:
                    property_row("Stop below [dB]", self.ui_scale)
                    changed, self.cfg.radio_map.stop_threshold_db = psim.SliderFloat(
                        "##rm_stop_db",
                        self.cfg.radio_map.stop_threshold_db,
                        -120.0,
                        -10.0,
                        format="%.0f",
                    )
                    end_property_row()
                    needs_update |= changed

                property_row("Roulette from depth", self.ui_scale)
                changed, self.cfg.radio_map.rr_depth = psim.SliderInt(
                    "##rm_rr_depth", self.cfg.radio_map.rr_depth, -1, 10
                )
                end_property_row()
                needs_update |= changed
                if psim.IsItemHovered():
                    psim.SetTooltip(
                        "Beyond this depth, rays are terminated at random to keep\n"
                        "the estimate unbiased for less work. -1 disables it."
                    )

                property_row("Survival probability", self.ui_scale)
                changed, self.cfg.radio_map.rr_prob = psim.SliderFloat(
                    "##rm_rr_prob", self.cfg.radio_map.rr_prob, 0.1, 0.99, format="%.2f"
                )
                end_property_row()
                needs_update |= changed
                psim.TreePop()

            property_row("Frame time target [ms]", self.ui_scale)
            changed, target_ms = psim.SliderFloat(
                "##rm_frame_budget",
                self.frame_time_target_s * 1000.0,
                16.0,
                250.0,
                format="%.0f",
            )
            end_property_row()
            if changed:
                self.frame_time_target_s = target_ms / 1000.0
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "How long a frame may take while the radio map refines.\n"
                    "Lower keeps the interface responsive; higher lets the\n"
                    "solver work in bigger batches, which converges faster."
                )

            property_row("Samples per iteration", self.ui_scale)
            changed, self.cfg.radio_map.log_samples_per_it = psim.SliderFloat(
                "##rm_samples",
                self.cfg.radio_map.log_samples_per_it,
                v_min=0,
                v_max=9,
                format="10^%.1f",
            )
            end_property_row()
            needs_update |= changed

            property_row("Max depth", self.ui_scale)
            changed, self.cfg.radio_map.max_depth = psim.SliderInt(
                "##rm_max_depth",
                self.cfg.radio_map.max_depth,
                v_min=1,
                v_max=10,
            )
            end_property_row()
            needs_update |= changed

            # -- Checkboxes table
            needs_update |= self._gui_features_checkboxes(self.cfg.radio_map, "##rm")

            # -- Radio map display options
            if self.radio_map is not None:
                psim.Spacing()

                psim.Text("Accumulating samples:")

                psim.PushStyleColor(psim.ImGuiCol_PlotHistogram, NVIDIA_GREEN)
                psim.ProgressBar(
                    min(
                        self.rm_accumulated_samples
                        / self.cfg.radio_map.accumulate_max_samples_per_tx,
                        1.0,
                    ),
                    (psim.CalcItemWidth(), 0),
                )
                psim.PopStyleColor()

                struct = ps.get_surface_mesh("radio_map")
                changed, show_rm = psim.Checkbox("Show radio map", struct.is_enabled())
                if changed:
                    struct.set_enabled(show_rm)

                psim.SameLine()
                _, self.cfg.radio_map.show_colorbar = psim.Checkbox(
                    "Show color bar", self.cfg.radio_map.show_colorbar
                )

                changed_cmap, self.rm_color_map_index = psim.Combo(
                    "Colormap",
                    self.rm_color_map_index,
                    self.rm_color_map_options,
                )
                if changed_cmap:
                    self.cfg.radio_map.color_map = self.rm_color_map_options[
                        self.rm_color_map_index
                    ]

                changed_vmin, self.cfg.radio_map.vmin = psim.SliderFloat(
                    "vmin",
                    self.cfg.radio_map.vmin,
                    v_min=-200,
                    v_max=0,
                )
                changed_vmax, self.cfg.radio_map.vmax = psim.SliderFloat(
                    "vmax",
                    self.cfg.radio_map.vmax,
                    v_min=-200,
                    v_max=0,
                )
                self.cfg.radio_map.vmin = min(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )
                self.cfg.radio_map.vmax = max(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )

                needs_visual_update = changed_cmap or changed_vmin or changed_vmax

            if self.cfg.radio_map.auto_update and needs_update:
                self.set_radio_map(self.compute_radio_map(), show=False)
            if needs_update or needs_visual_update:
                self.update_radio_map_colorbar()
                add_radio_map_to_polyscope(
                    "radio_map",
                    self.radio_map,
                    self.ps_groups,
                    self.cfg.radio_map,
                    direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                    use_alpha=self.cfg.radio_map.use_alpha,
                )

            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: some way to move the radio map (at least the plane z offset)

            psim.Spacing()

    def section_paths(self) -> None:
        if psim.CollapsingHeader("Paths", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False

            clicked = psim.Button("Compute paths")
            if clicked:
                self.update_paths(show=True)

            psim.SameLine()
            psim.BeginDisabled(self.paths is None)
            if psim.Button("Remove##paths"):
                self.clear_paths()
            psim.EndDisabled()

            psim.SameLine()
            _, self.cfg.paths.auto_update = psim.Checkbox(
                "Automatic update##paths", self.cfg.paths.auto_update
            )

            changed, self.cfg.paths.compute_cir = psim.Checkbox(
                "Channel impulse response##paths", self.cfg.paths.compute_cir
            )
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "Opens a window with the channel impulse response h:\n"
                    "one stem per propagation path, |h| in dB over delay."
                )
            if changed and self.cfg.paths.compute_cir:
                # Fill in the CIR data right away
                self.update_paths(show=True)

            property_row("Update at least every [ms]", self.ui_scale)
            changed, delay_ms = psim.SliderFloat(
                "##paths_update_interval",
                self.solver_delay_max_s * 1000.0,
                20.0,
                500.0,
                format="%.0f",
            )
            end_property_row()
            if changed:
                self.solver_delay_max_s = delay_ms / 1000.0
                self.solver_update_delay_s = min(
                    self.solver_update_delay_s, self.solver_delay_max_s
                )
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "How stale the paths may get while devices move.\n"
                    "Lower follows movement more closely and costs frames."
                )

            if psim.TreeNodeEx("Solver##paths_solver"):
                property_row("Samples per source", self.ui_scale)
                changed, log_samples = psim.SliderFloat(
                    "##paths_samples",
                    float(np.log10(max(self.cfg.paths.samples_per_src, 1))),
                    3.0,
                    7.0,
                    format="10^%.1f",
                )
                end_property_row()
                if changed:
                    self.cfg.paths.samples_per_src = int(10.0**log_samples)
                if psim.IsItemHovered():
                    psim.SetTooltip(
                        "Rays shot per source. More samples find more weak\n"
                        "diffuse paths; the dominant ones are found either way."
                    )

                property_row("While moving", self.ui_scale)
                changed, log_interactive = psim.SliderFloat(
                    "##paths_interactive_samples",
                    float(np.log10(max(self.cfg.paths.interactive_samples_per_src, 1))),
                    3.0,
                    7.0,
                    format="10^%.1f",
                )
                end_property_row()
                if changed:
                    self.cfg.paths.interactive_samples_per_src = int(
                        10.0**log_interactive
                    )
                if psim.IsItemHovered():
                    psim.SetTooltip(
                        "Samples used while devices move. A full-quality trace\n"
                        "follows once movement stops."
                    )

                property_row("Max paths per source", self.ui_scale)
                changed, log_paths = psim.SliderFloat(
                    "##paths_max_paths",
                    float(np.log10(max(self.cfg.paths.max_num_paths_per_src, 1))),
                    3.0,
                    7.0,
                    format="10^%.1f",
                )
                end_property_row()
                if changed:
                    self.cfg.paths.max_num_paths_per_src = int(10.0**log_paths)

                property_row("Seed", self.ui_scale)
                changed, self.cfg.paths.seed = psim.InputInt(
                    "##paths_seed", self.cfg.paths.seed
                )
                end_property_row()
                if psim.IsItemHovered():
                    psim.SetTooltip("Fixed, so repeated traces are reproducible.")
                psim.TreePop()

            property_row("Max depth", self.ui_scale)
            changed, self.cfg.paths.max_depth = psim.SliderInt(
                "##paths_max_depth",
                self.cfg.paths.max_depth,
                v_min=1,
                v_max=10,
            )
            end_property_row()
            needs_update |= changed

            psim.SetNextItemWidth(200 * self.ui_scale)
            changed, self.cfg.paths.synthetic_array = psim.Checkbox(
                "Synthetic array##paths", self.cfg.paths.synthetic_array
            )
            needs_update |= changed

            needs_update |= self._gui_features_checkboxes(self.cfg.paths, "##paths")

            # TODO: rendering parameters (radius, transparency, etc)
            # TODO: show & configure segment color per type of interaction
            needs_visual_update = False

            if self.cfg.paths.auto_update and needs_update:
                self.update_paths(show=False)
            if needs_update or needs_visual_update:
                add_paths_to_polyscope(self, self.paths, self.ps_groups)

            psim.Spacing()

    def section_animation(self) -> None:
        if psim.CollapsingHeader("Animation"):
            animation_gui(self)
            psim.Spacing()

    def section_rendering(self) -> None:
        if psim.CollapsingHeader("Rendering"):
            psim.Spacing()

            changed, combo_i = psim.Combo(
                "Rendering mode",
                self.cfg.rendering.mode.value,
                RENDERING_MODE_NAMES,
            )
            if changed:
                self.cfg.rendering.mode = RenderingMode(combo_i)
                self.set_rendering_mode(self.cfg.rendering.mode)

            if (self.cfg.rendering.mode == RenderingMode.RAY_TRACING) and (
                "cuda" in mi.variant()
            ):
                _, self.cfg.rendering.spp_per_frame = psim.SliderInt(
                    "SPP / frame",
                    self.cfg.rendering.spp_per_frame,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.spp_per_frame = max(
                    self.cfg.rendering.spp_per_frame, 1
                )
                _, self.cfg.rendering.max_accumulated_spp = psim.SliderInt(
                    "SPP max",
                    self.cfg.rendering.max_accumulated_spp,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.max_accumulated_spp = max(
                    self.cfg.rendering.max_accumulated_spp, 1
                )

                changed, self.cfg.rendering.relative_resolution = psim.SliderFloat(
                    "Rel. resolution",
                    self.cfg.rendering.relative_resolution,
                    v_min=0.1,
                    v_max=1.0,
                    format="%.2f",
                )
                self.cfg.rendering.relative_resolution = min(
                    max(self.cfg.rendering.relative_resolution, 0.1), 1.0
                )
                if changed:
                    self.clear_ray_traced_image()
                    # Need to re-create the denoiser for the new resolution
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

                changed, self.cfg.rendering.envmap_rotation_deg = psim.SliderFloat(
                    "Lighting angle",
                    self.cfg.rendering.envmap_rotation_deg,
                    v_min=-180,
                    v_max=180,
                    format="%.0f",
                )
                self.cfg.rendering.envmap_rotation_deg = min(
                    max(self.cfg.rendering.envmap_rotation_deg, -180), 180
                )
                if changed:
                    set_envmap_rotation(
                        self.render_cache, self.cfg.rendering.envmap_rotation_deg
                    )
                    self.reset_accumulation_requested = True

                changed, self.cfg.rendering.use_denoiser = psim.Checkbox(
                    "Use OptiX denoiser", self.cfg.rendering.use_denoiser
                )
                if changed:
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

            changed, self.cfg.use_vsync = psim.Checkbox("VSync", self.cfg.use_vsync)
            if changed:
                ps.set_enable_vsync(self.cfg.use_vsync)

            psim.SameLine()

            self.cfg.background_color = ps.get_background_color()
            changed, self.cfg.background_color = psim.ColorEdit4(
                "Background",
                self.cfg.background_color,
                psim.ImGuiColorEditFlags_NoInputs,
            )
            if changed:
                ps.set_background_color(self.cfg.background_color)

            changed, self.cfg.show_polyscope_gui = psim.Checkbox(
                "Show Polyscope UI", self.cfg.show_polyscope_gui
            )
            if changed:
                ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)

            psim.Spacing()

            if self.slice_plane is not None:
                psim.SeparatorText("Slice plane")
                changed, plane_active = psim.Checkbox(
                    "Active", self.slice_plane.get_active()
                )
                if changed:
                    self.set_slice_plane_active(plane_active)

                if plane_active:
                    psim.SameLine()
                    changed, draw_plane = psim.Checkbox(
                        "Show plane", self.slice_plane.get_draw_plane()
                    )
                    if changed:
                        self.slice_plane.set_draw_plane(draw_plane)

                    psim.SameLine()
                    changed, gizmo_active = psim.Checkbox(
                        "Show gizmo", self.slice_plane.get_draw_widget()
                    )
                    if changed:
                        self.slice_plane.set_draw_widget(gizmo_active)

    # ------------------------
    # Docked workspace layout

    def start_simulation(self) -> None:
        """Begin (or resume) animating devices and refining the radio map."""
        first_start = not self.simulation_started
        self.simulation_running = True
        self.simulation_started = True
        self.animation_config.playing = True
        self.animation_config.time_started = time.time()
        if first_start:
            # A radio map is only built when the scenario asked for one: the
            # radio map's auto-update means "refresh the existing map", not
            # "create one", and computing it unasked is expensive.
            if (
                self.compute_radio_map_on_start
                and self.radio_map is None
                and self.scene._transmitters
            ):
                self.set_radio_map(self.compute_radio_map(), show=True)
            if self.cfg.paths.auto_update and self.paths is None:
                self.update_paths(show=True)

    def pause_simulation(self) -> None:
        self.simulation_running = False
        self.animation_config.playing = False

    def simulation_button(self, scale: float) -> None:
        """
        Start / Pause / Resume, as one button whose colour says what pressing
        it will do: green to run, amber to hold.
        """
        if self.simulation_running:
            label, color, hovered = "Pause", (0.72, 0.52, 0.16, 1.0), (0.80, 0.60, 0.20, 1.0)
        else:
            label = "Resume" if self.simulation_started else "Start"
            color, hovered = (0.30, 0.52, 0.24, 1.0), (0.36, 0.60, 0.28, 1.0)

        psim.PushStyleColor(psim.ImGuiCol_Button, color)
        psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, hovered)
        pressed = psim.Button(f"{label}##simulation", (76 * scale, 0))
        psim.PopStyleColor(2)
        if psim.IsItemHovered():
            psim.SetTooltip(
                "Pause the animation and stop refining the radio map"
                if self.simulation_running
                else "Animate the devices and compute the radio results"
            )
        if pressed:
            self.pause_simulation() if self.simulation_running else self.start_simulation()

    def set_docked_layout(self, docked: bool) -> None:
        """Switch between the docked workspace and floating windows."""
        if self.cfg.use_docked_layout == docked:
            return
        self.cfg.use_docked_layout = docked
        if docked:
            set_editor_imgui_style()
        else:
            set_custom_imgui_style()

    def workspace_gui(self) -> None:
        """
        Fixed, non-overlapping areas: a top bar, a tool column, the outliner
        and properties on the side, a timeline, and a status bar. The 3D view
        is whatever these do not cover.
        """
        scale = self.ui_scale
        areas = self.layout.areas(ps.get_window_size(), scale)

        self._workspace_topbar(areas["topbar"], scale)
        self._workspace_tools(areas["tools"], scale)
        self._workspace_outliner(areas["outliner"], scale)
        self._workspace_properties(areas["properties"], scale)
        self._workspace_timeline(areas["timeline"], scale)
        self._workspace_status(areas["status"], scale)

        # Draggable borders between the areas
        tools = areas["tools"]
        self.layout.tools_width += splitter(
            "tools", (tools[0] + tools[2], tools[1], tools[2], tools[3]), True, scale
        )
        side = areas["outliner"]
        self.layout.side_width -= splitter(
            "side", (side[0], side[1], side[2], side[3] + areas["properties"][3]), True, scale
        )
        self.layout.outliner_fraction += splitter(
            "outliner", (side[0], side[1] + side[3], side[2], side[3]), False, scale
        ) * scale / max(areas["properties"][3] + side[3], 1.0)
        timeline = areas["timeline"]
        self.layout.timeline_height -= splitter(
            "timeline", (timeline[0], timeline[1], timeline[2], timeline[3]), False, scale
        )

    def _workspace_topbar(self, rect, scale: float) -> None:
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (8 * scale, 3 * scale))
        if begin_area("##topbar", rect, background=HEADER_BG):
            psim.AlignTextToFramePadding()
            psim.TextColored((*ACCENT_BRIGHT, 1.0), "SIONNA RT")
            psim.SameLine()
            psim.TextDisabled("|")
            psim.SameLine()

            psim.PushItemWidth(190 * scale)
            changed, combo_i = psim.Combo(
                "##topbar_scene", self.current_scene_idx, self.known_scene_names
            )
            psim.PopItemWidth()
            if changed:
                self.load_scene_requested = self.known_scene_paths[combo_i]

            psim.SameLine()
            psim.PushItemWidth(150 * scale)
            changed, mode_i = psim.Combo(
                "##topbar_mode",
                self.cfg.rendering.mode.value,
                RENDERING_MODE_NAMES,
            )
            psim.PopItemWidth()
            if changed:
                self.set_rendering_mode(RenderingMode(mode_i))

            psim.SameLine()
            if psim.Button("Top##topbar"):
                self.move_camera_top()
            psim.SameLine()
            if psim.Button("Fit##topbar"):
                self.fit_camera_to_scene()
            psim.SameLine()
            if psim.Button("Home##topbar"):
                self.move_camera_home()

            # Right-aligned actions, sized from their actual labels so the
            # last one is never clipped
            labels = ["Save view", "?"]
            style = psim.GetStyle()
            widths = [
                psim.CalcTextSize(label)[0] + 2 * style.FramePadding[0]
                for label in labels
            ]
            buttons_width = sum(widths) + (len(labels) - 1) * style.ItemSpacing[0]
            psim.SameLine()
            psim.SetCursorPosX(
                max(
                    psim.GetCursorPosX(),
                    psim.GetCursorPosX()
                    + psim.GetContentRegionAvail()[0]
                    - buttons_width
                    - style.WindowPadding[0],
                )
            )
            if psim.Button(f"{labels[0]}##topbar"):
                self.save_screenshot()
            if psim.IsItemHovered():
                psim.SetTooltip("Save a PNG screenshot of the current view")
            psim.SameLine()
            if psim.Button(f"{labels[1]}##topbar"):
                self.cfg.show_help_window = not self.cfg.show_help_window
            if psim.IsItemHovered():
                psim.SetTooltip("Controls & shortcuts (H)")
        end_area()
        psim.PopStyleVar()

    def _workspace_tools(self, rect, scale: float) -> None:
        if rect[2] < 8 * scale:
            return
        if begin_area("##tools", rect):
            area_header("Add", scale)
            button = 34.0 * scale
            per_row = max(int(psim.GetContentRegionAvail()[0] // (button + 6 * scale)), 1)

            entries = [
                ("tx", icons.transmitter, "Transmitter", None),
                ("rx", icons.receiver, "Receiver", None),
            ]
            for spec in ASSET_LIBRARY:
                entries.append(
                    (
                        spec.key,
                        icons.ASSET_ICONS.get(spec.key, icons.object_icon),
                        f"{spec.label}\n{spec.note}",
                        spec,
                    )
                )

            for index, (key, draw, tooltip, spec) in enumerate(entries):
                if index % per_row != 0:
                    psim.SameLine()
                if icons.icon_button(f"tools_{key}", draw, button, scale, tooltip):
                    if key == "tx":
                        self.add_radio_device(
                            self.default_new_device_position(True), is_transmitter=True
                        )
                    elif key == "rx":
                        self.add_radio_device(
                            self.default_new_device_position(False), is_transmitter=False
                        )
                    else:
                        self.add_asset(spec)
            psim.Dummy((0.0, 4 * scale))

            psim.Dummy((0.0, 6 * scale))
            area_header("Show", scale)
            for label, key in (
                ("Radio map", "radio_maps"),
                ("Paths", "paths"),
                ("Devices", "rd"),
                ("Scene", "scene"),
            ):
                group = self.ps_groups.get(key)
                if group is None:
                    continue
                names = list(group.get_child_structure_names())
                visible = any(
                    struct.is_enabled()
                    for struct in (self._find_structure(n) for n in names)
                    if struct is not None
                )
                changed, visible = psim.Checkbox(f"{label}##show_{key}", visible)
                if changed:
                    group.set_enabled(visible)
                    for name in names:
                        struct = self._find_structure(name)
                        if struct is not None:
                            struct.set_enabled(visible)
            if self.slice_plane is not None:
                changed, active = psim.Checkbox(
                    "Slice plane##show", self.slice_plane.get_active()
                )
                if changed:
                    self.set_slice_plane_active(active)
        end_area()

    @staticmethod
    def _find_structure(name: str):
        """Look up a Polyscope structure by name, whatever its type."""
        for has, get in (
            (ps.has_surface_mesh, ps.get_surface_mesh),
            (ps.has_point_cloud, ps.get_point_cloud),
            (ps.has_curve_network, ps.get_curve_network),
        ):
            if has(name):
                return get(name)
        return None

    def _workspace_outliner(self, rect, scale: float) -> None:
        """
        Scene tree: radio devices (clickable to select), placed assets and the
        Polyscope structure groups with their visibility toggles.
        """
        if begin_area("##outliner", rect):
            area_header("Outliner", scale)

            for is_transmitter, devices in (
                (True, self.scene._transmitters),
                (False, self.scene._receivers),
            ):
                for name, rd in devices.items():
                    psim.PushID(f"outliner_{name}")
                    swatch = psim.GetFontSize() * 0.85
                    psim.ColorButton(
                        "##c",
                        (*rd.color, 1.0),
                        psim.ImGuiColorEditFlags_NoTooltip,
                        (swatch, swatch),
                    )
                    psim.SameLine()
                    if psim.Selectable(name, self.selected_object is rd):
                        self.selected_object = rd
                        self.selected_type = (
                            SelectionType.Transmitter
                            if is_transmitter
                            else SelectionType.Receiver
                        )
                        self.properties_tab = PROPERTIES_TABS.index("Object")
                    psim.PopID()

            for asset_name in self.placed_assets:
                psim.PushID(f"outliner_asset_{asset_name}")
                is_selected = (
                    self.selected_type == SelectionType.Mesh
                    and getattr(self.selected_object, "name", None) == asset_name
                )
                if psim.Selectable(asset_name, is_selected):
                    self.select_scene_object(asset_name)
                psim.PopID()

            psim.Dummy((0.0, 4 * scale))
            psim.Separator()
            for label, key in (
                ("Scene meshes", "scene"),
                ("Radio maps", "radio_maps"),
                ("Paths", "paths"),
                ("Radio devices", "rd"),
            ):
                group = self.ps_groups.get(key)
                if group is None:
                    continue
                children = list(group.get_child_structure_names())
                if not psim.TreeNodeEx(f"{label} ({len(children)})##outliner_{key}"):
                    continue
                for child in children:
                    struct = self._find_structure(child)
                    if struct is None:
                        psim.TextDisabled(child)
                        continue
                    psim.PushID(f"outliner_child_{child}")
                    changed, enabled = psim.Checkbox("", struct.is_enabled())
                    if changed:
                        struct.set_enabled(enabled)
                    psim.SameLine()
                    if psim.TreeNodeEx(child):
                        changed, color = psim.ColorEdit3(
                            "Color##outliner",
                            struct.get_color(),
                            psim.ImGuiColorEditFlags_NoInputs,
                        )
                        if changed:
                            struct.set_color(color)
                        psim.PushItemWidth(-90 * scale)
                        changed, transparency = psim.SliderFloat(
                            "Opacity##outliner", struct.get_transparency(), 0.05, 1.0
                        )
                        if changed:
                            struct.set_transparency(transparency)
                        if ps.has_point_cloud(child):
                            changed, radius = psim.SliderFloat(
                                "Radius##outliner",
                                struct.get_radius(relative=False),
                                0.1,
                                20.0,
                            )
                            if changed:
                                struct.set_radius(radius, relative=False)
                        psim.PopItemWidth()
                        psim.TreePop()
                    psim.PopID()
                psim.TreePop()

            psim.Dummy((0.0, 4 * scale))
            psim.PushStyleColor(psim.ImGuiCol_Text, (0.45, 0.45, 0.45, 1.0))
            changed, self.cfg.show_polyscope_gui = psim.Checkbox(
                "Advanced: viewer panels", self.cfg.show_polyscope_gui
            )
            psim.PopStyleColor()
            if psim.IsItemHovered():
                psim.SetTooltip(
                    "Opens the viewer's own windows on top of this layout.\n"
                    "They place themselves and will overlap the panels, so\n"
                    "only turn this on for a control missing from here."
                )
            if changed:
                ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)
        end_area()

    def _workspace_properties(self, rect, scale: float) -> None:
        if begin_area("##properties", rect):
            area_header("Properties", scale)
            button = 30.0 * scale
            for index, name in enumerate(PROPERTIES_TABS):
                if index > 0:
                    psim.SameLine()
                if icons.icon_button(
                    f"proptab_{name}",
                    PROPERTIES_TAB_ICONS[name],
                    button,
                    scale,
                    name,
                    active=index == self.properties_tab,
                ):
                    self.properties_tab = index
            psim.Dummy((0.0, 2 * scale))
            psim.Separator()
            psim.TextDisabled(PROPERTIES_TABS[self.properties_tab].upper())
            psim.Dummy((0.0, 2 * scale))
            match PROPERTIES_TABS[self.properties_tab]:
                case "Object":
                    selection_contents(self, self.selected_object, self.selected_type)
                case "Devices":
                    self.section_devices()
                case "Radio map":
                    self.section_radio_map()
                case "Paths":
                    self.section_paths()
                case "Analysis":
                    if psim.CollapsingHeader(
                        "Link budget", psim.ImGuiTreeNodeFlags_DefaultOpen
                    ):
                        link_budget_contents(self)
                    if psim.CollapsingHeader(
                        "Channel quality", psim.ImGuiTreeNodeFlags_DefaultOpen
                    ):
                        link_simulation_contents(self)
                    if psim.CollapsingHeader(
                        "Coverage", psim.ImGuiTreeNodeFlags_DefaultOpen
                    ):
                        coverage_contents(self)

                case "Assets":
                    self.section_assets()
                case "Scene":
                    # The scene picker and camera presets live in the top bar
                    self.section_scene(include_picker=False, include_camera=False)
                case "Render":
                    self.section_rendering()
        end_area()

    def invalidate_link_metrics(self) -> None:
        """
        Forget measured link results. They describe one particular geometry, so
        once devices move back to the start of their paths they no longer
        describe anything in the scene.
        """
        self.phy_metrics_stats = None
        self.nr_link_stats = None
        self.link_metrics_time = 0.0
        # Stay cleared until asked again: measuring straight away would put the
        # numbers back on screen and look as though nothing had been reset.
        self.phy_auto_measure = False

    def transport_gui(self, scale: float) -> None:
        """
        Start / pause, restart and speed, compact enough to live in the bottom
        area's header so they stay reachable whichever editor is open.
        """
        self.simulation_button(scale)
        psim.SameLine()
        if psim.Button("Restart##transport"):
            restart_trajectories(self)
        if psim.IsItemHovered():
            psim.SetTooltip("Send every animated device back to the start of its path")
        psim.SameLine()
        psim.PushItemWidth(78 * scale)
        speeds = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        labels = [f"{speed:g}x" for speed in speeds]
        current = min(
            range(len(speeds)),
            key=lambda i: abs(speeds[i] - self.animation_config.speed_multiplier),
        )
        changed, chosen = psim.Combo("##transport_speed", current, labels)
        psim.PopItemWidth()
        if changed:
            self.animation_config.speed_multiplier = speeds[chosen]
        if psim.IsItemHovered():
            psim.SetTooltip("How fast animated devices move")

    def _workspace_timeline(self, rect, scale: float) -> None:
        if rect[3] < 8 * scale:
            return
        if begin_area("##timeline", rect):
            previous_tab = self.bottom_tab
            button = 28.0 * scale
            for index, name in enumerate(BOTTOM_TABS):
                if index > 0:
                    psim.SameLine()
                if icons.icon_button(
                    f"bottomtab_{name}",
                    BOTTOM_TAB_ICONS[name],
                    button,
                    scale,
                    name,
                    active=index == self.bottom_tab,
                ):
                    self.bottom_tab = index
            psim.SameLine()
            psim.AlignTextToFramePadding()
            psim.TextDisabled(BOTTOM_TABS[self.bottom_tab].upper())

            # The transport belongs to the whole application, not to one editor
            style = psim.GetStyle()
            transport_width = (
                76 * scale
                + psim.CalcTextSize("Restart")[0]
                + 2 * style.FramePadding[0]
                + 78 * scale
                + 2 * style.ItemSpacing[0]
            )
            psim.SameLine()
            psim.SetCursorPosX(
                max(
                    psim.GetCursorPosX(),
                    psim.GetCursorPosX()
                    + psim.GetContentRegionAvail()[0]
                    - transport_width
                    - style.WindowPadding[0],
                )
            )
            self.transport_gui(scale)
            if self.bottom_tab != previous_tab and BOTTOM_TABS[self.bottom_tab] != "Timeline":
                # The plots need more room than the transport controls
                self.layout.timeline_height = max(self.layout.timeline_height, 300.0)
            psim.Dummy((0.0, 2 * scale))

            match BOTTOM_TABS[self.bottom_tab]:
                case "Impulse response":
                    cir_contents(self)
                    end_area()
                    return
                case "Link":
                    phy_link_contents(self)
                    end_area()
                    return
                case "Antenna pattern":
                    if self.selected_object is not None and self.selected_type in (
                        SelectionType.Transmitter,
                        SelectionType.Receiver,
                    ):
                        pattern_cuts_contents(
                            self,
                            self.scene.tx_array
                            if self.selected_type == SelectionType.Transmitter
                            else self.scene.rx_array,
                        )
                    else:
                        psim.TextDisabled(
                            "Select a radio device to see its antenna pattern."
                        )
                    end_area()
                    return

            psim.Dummy((0.0, 2 * scale))
            # Scrub the selected device's trajectory, when it has one
            if self.selected_object is not None and self.selected_type in (
                SelectionType.Transmitter,
                SelectionType.Receiver,
            ):
                trajectory = self.animation_config.trajectories.get(
                    self.selected_object.name
                )
                if trajectory is not None and len(trajectory) > 0:
                    psim.SameLine()
                    psim.PushItemWidth(-120 * scale)
                    changed, distance = psim.SliderFloat(
                        f"{self.selected_object.name}##timeline_scrub",
                        trajectory.distance,
                        0.0,
                        trajectory.total_distance(),
                        format="%.1f m",
                    )
                    psim.PopItemWidth()
                    if changed:
                        trajectory.enabled = False
                        trajectory.backward = False
                        trajectory.distance = distance
                        if apply_trajectory_position(self.selected_object, trajectory):
                            self.update_attached_object(self.selected_object)
                            propagate_device_updates(
                                self,
                                self.selected_type == SelectionType.Transmitter,
                                self.selected_type == SelectionType.Receiver,
                            )
        end_area()

    def _workspace_status(self, rect, scale: float) -> None:
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (8 * scale, 2 * scale))
        if begin_area("##status", rect, background=HEADER_BG):
            io = psim.GetIO()
            n_meshes, n_triangles = getattr(self, "scene_stats", (0, 0))
            frequency_ghz = float(self.scene.frequency[0]) / 1e9
            note = getattr(self, "last_export_note", None)
            if n_triangles >= 1_000_000:
                triangles = f"{n_triangles / 1e6:.1f}M"
            elif n_triangles >= 10_000:
                triangles = f"{n_triangles / 1e3:.0f}k"
            else:
                triangles = str(n_triangles)
            parts = [
                f"{io.Framerate:.0f} FPS",
                f"{len(self.scene._transmitters)} TX, {len(self.scene._receivers)} RX",
                f"{n_meshes} meshes, {triangles} tris",
                f"{frequency_ghz:.3g} GHz",
            ]
            psim.TextDisabled("   |   ".join(parts))
            if note is not None and time.time() - note[1] < 8.0:
                psim.SameLine()
                psim.TextColored((*ACCENT_BRIGHT, 1.0), f"   {note[0]}")
        end_area()
        psim.PopStyleVar()

    def gui(self):
        # TODO: change GUI accent color to a non-default color.

        # Highlight the object under the cursor while picking one
        self.update_attach_highlight()

        if self.cfg.gui_mode == GuiMode.HIDDEN:
            # Tab hides the interface, which is useful for a clean view but
            # looks like a failure without a way back on screen.
            scale = self.ui_scale
            psim.GetForegroundDrawList().AddText(
                (12 * scale, 10 * scale),
                im_col32(0.62, 0.64, 0.65, 0.85),
                "Interface hidden - press Tab to show it",
            )
            return

        self.viewport_context_menu()

        if self.cfg.use_docked_layout:
            self.workspace_gui()
            if self.cfg.show_help_window:
                self.gui_help_window()
            return

        # --- Selection window
        if self.selected_object is not None:
            selection_gui(self, self.selected_object, self.selected_type)

        # --- Antenna pattern cuts (needs a selected radio device)
        if self.selected_object is not None and self.selected_type in (
            SelectionType.Transmitter,
            SelectionType.Receiver,
        ):
            array = (
                self.scene.tx_array
                if self.selected_type == SelectionType.Transmitter
                else self.scene.rx_array
            )
            pattern_cuts_window(self, array)

        # --- Channel impulse response
        cir_window(self)

        # --- Help window
        if self.cfg.show_help_window:
            self.gui_help_window()

        # --- Colorbar window
        if (
            self.has_visible_radio_map()[0]
            and self.cfg.radio_map.show_colorbar
            and (self.rm_colorbar is not None)
            and (self.rm_colorbar_texture_id is not None)
            and hasattr(psim, "Image")
        ):
            window_resolution = ps.get_window_size()
            h, w = self.rm_colorbar.shape[:2]
            psim.SetNextWindowSize((w * self.ui_scale, h * self.ui_scale))
            psim.SetNextWindowPos(
                (0.5 * (window_resolution[0] - w * self.ui_scale), 5 * self.ui_scale)
            )
            psim.Begin(
                "Colorbar",
                open=True,
                flags=(
                    psim.ImGuiWindowFlags_NoTitleBar
                    | psim.ImGuiWindowFlags_NoDecoration
                    | psim.ImGuiWindowFlags_NoBackground
                ),
            )
            psim.SetCursorPosX(0)
            psim.SetCursorPosY(0)
            psim.Image(
                psim.ImTextureRef(self.rm_colorbar_texture_id),
                (self.rm_colorbar.shape[1], self.rm_colorbar.shape[0]),
            )
            psim.End()

        # --- Main GUI window
        psim.SetNextWindowSize(
            (430 * self.ui_scale, 800 * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (10 * self.ui_scale, 10 * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.Begin("Sionna RT##sionna", open=True)

        self.header_gui()

        self.section_scene()
        self.section_assets()
        self.section_devices()
        self.section_radio_map()
        self.section_paths()
        self.section_animation()
        self.section_rendering()
        psim.End()  # End main Sionna RT window

    def _gui_features_checkboxes(
        self, cfg: RadioMapConfig | PathsConfig, suffix: str
    ) -> bool:
        any_changed = False

        psim.Columns(2, borders=False)
        psim.SetColumnWidth(0, 165 * self.ui_scale)

        changed, cfg.los = psim.Checkbox("Line of sight" + suffix, cfg.los)
        any_changed |= changed

        changed, cfg.diffuse_reflection = psim.Checkbox(
            "Diffuse reflection" + suffix, cfg.diffuse_reflection
        )
        any_changed |= changed

        changed, cfg.diffraction = psim.Checkbox(
            "Diffraction" + suffix, cfg.diffraction
        )
        any_changed |= changed

        psim.NextColumn()

        changed, cfg.specular_reflection = psim.Checkbox(
            "Specular reflection" + suffix, cfg.specular_reflection
        )
        any_changed |= changed

        changed, cfg.refraction = psim.Checkbox("Refraction" + suffix, cfg.refraction)
        any_changed |= changed

        if cfg.diffraction:
            changed, cfg.edge_diffraction = psim.Checkbox(
                "Edge" + suffix, cfg.edge_diffraction
            )
            any_changed |= changed

            psim.SameLine()
            changed, cfg.diffraction_lit_region = psim.Checkbox(
                "Lit region" + suffix, cfg.diffraction_lit_region
            )
            any_changed |= changed

        psim.Columns(1)

        return any_changed

    def gui_help_window(self):
        window_resolution = ps.get_window_size()
        w, h = 600, 600
        psim.SetNextWindowSize(
            (w * self.ui_scale, h * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (
                0.5 * (window_resolution[0] - w * self.ui_scale),
                0.5 * (window_resolution[1] - h * self.ui_scale),
            ),
            psim.ImGuiCond_FirstUseEver,
        )

        # Note: ImGuiWindowFlags_Modal is an internal flag for popups. Passing it
        # to Begin makes it report the window as closed, which was written back
        # into the setting and switched help off a frame after opening it.
        _, keep_open = psim.Begin(
            "Controls & shortcuts###help_window",
            open=True,
            flags=psim.ImGuiWindowFlags_NoFocusOnAppearing,
        )
        self.cfg.show_help_window = keep_open

        psim.TextColored((*ACCENT_BRIGHT, 1.0), "S I O N N A   R T")
        psim.TextDisabled(
            f"v{GUI_VERSION[0]}.{GUI_VERSION[1]}.{GUI_VERSION[2]} - "
            "(c) NVIDIA Corporation 2025\n"
            "Uses map data from OpenStreetMap (openstreetmap.org/copyright)."
        )

        psim.NewLine()

        for title, table in HELP_WINDOW_TABLES.items():
            # Centered text
            text_pos = (
                psim.GetCursorPosX()
                + psim.GetColumnWidth(0) * 0.5
                - psim.CalcTextSize(title)[0]
            )
            psim.SetCursorPosX(text_pos * self.ui_scale)
            psim.TextColored((*ACCENT_BRIGHT, 1.0), title)

            psim.Separator()
            psim.Columns(2)
            psim.SetColumnWidth(0, 220 * self.ui_scale)
            psim.Text(os.linesep.join(table.keys()))
            psim.NextColumn()
            psim.Text(os.linesep.join(table.values()))
            psim.Columns(1)
            psim.Separator()

            psim.NewLine()

        psim.Columns(1)
        psim.End()
