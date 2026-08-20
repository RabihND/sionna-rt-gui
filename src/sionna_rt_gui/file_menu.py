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


class FileBrowser:
    """Minimal in-app browser for picking a scene XML file."""

    def __init__(self):
        self.visible: bool = False
        self.current_dir: str = os.path.expanduser("~")

    def open(self, start_dir: str | None = None) -> None:
        if start_dir and os.path.isdir(start_dir):
            self.current_dir = start_dir
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
        _, self.visible = psim.Begin("Open scene##file_browser", open=True)

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
        for entry in entries:
            if entry.startswith(".") or not entry.endswith(".xml"):
                continue
            full = os.path.join(self.current_dir, entry)
            if os.path.isfile(full):
                if _clicked(psim.Selectable(f"{entry}##fb_file_{entry}", False)):
                    app.load_scene_requested = full
                    self.visible = False
        psim.EndChild()

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
