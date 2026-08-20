#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
"File" menu for the top bar: open a scene from disk with an in-app file
browser, re-open recently used scenes, or load one of the built-in demo
scenes. The recent list is persisted to ``~/.sionna-rt-gui/recent_scenes.yaml``.
"""
from __future__ import annotations

import os
import shutil
import sys

import yaml

import polyscope as ps
import polyscope.imgui as psim

from .sionna_utils import get_built_in_scenes

RECENT_SCENES_PATH = os.path.expanduser(
    os.path.join("~", ".sionna-rt-gui", "recent_scenes.yaml")
)
MAX_RECENT_SCENES = 10


def _clicked(result) -> bool:
    """MenuItem/Selectable bindings may return a bool or a (clicked, ...) tuple."""
    return bool(result[0]) if isinstance(result, tuple) else bool(result)


def load_recent_scenes() -> list[str]:
    if not os.path.isfile(RECENT_SCENES_PATH):
        return []
    try:
        with open(RECENT_SCENES_PATH, "r") as f:
            loaded = yaml.safe_load(f)
        return [str(p) for p in (loaded or [])]
    except Exception as e:
        print(f"[!] Failed to load recent scenes: {e}", file=sys.stderr)
        return []


def save_recent_scenes(recent: list[str]) -> None:
    try:
        os.makedirs(os.path.dirname(RECENT_SCENES_PATH), exist_ok=True)
        with open(RECENT_SCENES_PATH, "w") as f:
            yaml.safe_dump(recent, f)
    except Exception as e:
        print(f"[!] Failed to save recent scenes: {e}", file=sys.stderr)


def add_recent_scene(path: str, recent: list[str]) -> list[str]:
    path = os.path.realpath(path)
    recent = [path] + [p for p in recent if p != path]
    recent = recent[:MAX_RECENT_SCENES]
    save_recent_scenes(recent)
    return recent


def save_scene_copy(app, dest_parent: str, name: str) -> str | None:
    """
    Copy the directory of the currently loaded scene (scene.xml + meshes)
    to ``dest_parent/name``. Returns the new scene XML path, or None.
    """
    src_xml = app.cfg.scene_filename
    if not (src_xml and os.path.isfile(src_xml)):
        app.addons.notify("No scene file to save.")
        return None
    name = name.strip() or "scene"
    src_dir = os.path.dirname(os.path.realpath(src_xml))
    dest_dir = os.path.join(dest_parent, name)
    if os.path.exists(dest_dir):
        app.addons.notify(f"Not saved: '{dest_dir}' already exists.")
        return None
    try:
        shutil.copytree(src_dir, dest_dir)
    except OSError as e:
        app.addons.notify(f"Failed to save scene: {e}")
        return None
    new_xml = os.path.join(dest_dir, os.path.basename(src_xml))
    app.cfg.scene_filename = new_xml
    app.recent_scenes = add_recent_scene(new_xml, app.recent_scenes)
    app.addons.notify(f"Scene saved to {dest_dir}")
    return new_xml


class FileBrowser:
    """Minimal in-app browser for picking a scene XML file or a save location."""

    def __init__(self):
        self.visible: bool = False
        self.current_dir: str = os.path.expanduser("~")
        self.save_mode: bool = False
        self.save_name: str = "scene"

    def open(self, start_dir: str | None = None) -> None:
        if start_dir and os.path.isdir(start_dir):
            self.current_dir = start_dir
        self.save_mode = False
        self.visible = True

    def open_save(self, default_name: str, start_dir: str | None = None) -> None:
        if start_dir and os.path.isdir(start_dir):
            self.current_dir = start_dir
        self.save_name = default_name
        self.save_mode = True
        self.visible = True

    def draw(self, app) -> None:
        if not self.visible:
            return
        scale = app.ui_scale
        window_resolution = ps.get_window_size()
        w, h = 560 * scale, 420 * scale
        psim.SetNextWindowSize((w, h), psim.ImGuiCond_FirstUseEver)
        psim.SetNextWindowPos(
            (0.5 * (window_resolution[0] - w), 0.5 * (window_resolution[1] - h)),
            psim.ImGuiCond_FirstUseEver,
        )
        title = "Save scene##file_browser" if self.save_mode else "Open scene##file_browser"
        _, self.visible = psim.Begin(title, open=True)

        changed, edited = psim.InputText("##file_browser_path", self.current_dir)
        if changed and os.path.isdir(os.path.expanduser(edited)):
            self.current_dir = os.path.expanduser(edited)
        psim.SameLine()
        if psim.Button("Home##file_browser"):
            self.current_dir = os.path.expanduser("~")

        psim.BeginChild("##file_browser_list", (0, -34 * scale))
        try:
            entries = sorted(os.listdir(self.current_dir), key=str.lower)
        except OSError:
            entries = []

        parent = os.path.dirname(self.current_dir.rstrip(os.sep))
        if parent and parent != self.current_dir:
            if _clicked(psim.Selectable("[..]##file_browser_up", False)):
                self.current_dir = parent

        for entry in entries:
            if entry.startswith("."):
                continue
            full = os.path.join(self.current_dir, entry)
            if os.path.isdir(full):
                if _clicked(psim.Selectable(f"[{entry}]##fb_dir_{entry}", False)):
                    self.current_dir = full
        if not self.save_mode:
            for entry in entries:
                if entry.startswith(".") or not entry.endswith(".xml"):
                    continue
                full = os.path.join(self.current_dir, entry)
                if os.path.isfile(full):
                    if _clicked(psim.Selectable(f"{entry}##fb_file_{entry}", False)):
                        app.load_scene_requested = full
                        self.visible = False
        psim.EndChild()

        if self.save_mode:
            psim.SetNextItemWidth(200 * scale)
            _, self.save_name = psim.InputText("##fb_save_name", self.save_name)
            psim.SameLine()
            if psim.Button("Save here##file_browser"):
                if save_scene_copy(app, self.current_dir, self.save_name):
                    self.visible = False
            psim.SameLine()
            psim.TextDisabled("Saves the scene folder under this directory.")
        else:
            psim.TextDisabled("Click a scene .xml file to load it.")
        psim.SameLine()
        if psim.Button("Cancel##file_browser"):
            self.visible = False

        psim.End()


def file_menu_gui(app) -> None:
    """The "File" top-bar button and its dropdown. Call once per frame."""
    if psim.Button("File##topbar"):
        psim.OpenPopup("##file_menu")

    if psim.BeginPopup("##file_menu"):
        if _clicked(psim.MenuItem("Open scene...")):
            start = None
            if app.recent_scenes:
                start = os.path.dirname(app.recent_scenes[0])
            app.file_browser.open(start)
            psim.CloseCurrentPopup()

        if _clicked(psim.MenuItem("Save scene as...")):
            current = app.cfg.scene_filename or ""
            default_name = (
                os.path.basename(os.path.dirname(current)) if current else "scene"
            )
            app.file_browser.open_save(default_name)
            psim.CloseCurrentPopup()

        if psim.BeginMenu("Open recent"):
            if not app.recent_scenes:
                psim.TextDisabled("(empty)")
            for path in list(app.recent_scenes):
                label = os.path.basename(path)
                if _clicked(psim.MenuItem(f"{label}##recent_{path}")):
                    app.load_scene_requested = path
                if psim.IsItemHovered():
                    psim.SetTooltip(path)
            if app.recent_scenes:
                psim.Separator()
                if _clicked(psim.MenuItem("Clear recent")):
                    app.recent_scenes = []
                    save_recent_scenes([])
            psim.EndMenu()

        if psim.BeginMenu("Demo scenes"):
            for name, path in get_built_in_scenes().items():
                if _clicked(psim.MenuItem(f"{name}##demo_{name}")):
                    app.load_scene_requested = path
            psim.EndMenu()

        psim.EndPopup()

    app.file_browser.draw(app)
