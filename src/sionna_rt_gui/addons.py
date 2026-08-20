#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Blender-style addon system.

An addon is a Python package exposing:

    ADDON_INFO = {"name": "My addon", "version": "0.1", "author": "...",
                  "description": "...", "min_gui_version": "0.1"}

    def register(api): ...
    def unregister(api): ...

Addons are discovered from two channels:

1. The user addons directory: ``~/.sionna-rt-gui/addons/<name>/``
2. pip entry points in the ``sionna_rt_gui.addons`` group.

Addons interact with the application exclusively through :class:`AddonAPI`.
Every addon callback is wrapped in try/except: an error notifies the user and
disables the addon, it never crashes the application. See ``ADDON_SPEC.md`` at
the repository root for the full contract.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field as dataclass_field
from types import ModuleType
from typing import Any, Callable

import polyscope as ps
import polyscope.imgui as psim
from omegaconf import OmegaConf

from . import __version__ as GUI_VERSION
from .config import GuiMode
from .reload import reload_module_recursive

API_VERSION = (1, 0)
ENTRY_POINT_GROUP = "sionna_rt_gui.addons"
USER_ADDONS_DIR = os.path.expanduser(os.path.join("~", ".sionna-rt-gui", "addons"))
# Enabled / disabled choices made in the GUI are persisted here (OmegaConf YAML)
# so that they survive application restarts.
ADDONS_STATE_PATH = os.path.expanduser(
    os.path.join("~", ".sionna-rt-gui", "addons.yaml")
)
NOTIFICATION_DURATION_S = 6.0


def parse_version(value: Any) -> tuple[int, ...]:
    """Parse ``"0.1"``, ``(0, 1)`` or ``[0, 1]`` into a tuple of ints."""
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split("."))


@dataclass
class Addon:
    name: str
    # Discovery channel: "user" (directory) or "pip" (entry point)
    source: str
    # Directory of the addon package, if known (always set for "user" addons).
    path: str | None = None
    entry_point: Any = None
    module: ModuleType | None = None
    info: dict = dataclass_field(default_factory=dict)
    api: "AddonAPI | None" = None
    enabled: bool = False
    error: str | None = None

    @property
    def display_name(self) -> str:
        name = str(self.info.get("name", self.name))
        version = self.info.get("version")
        return f"{name} {version}" if version else name


class AddonAPI:
    """
    The addon-facing API (v1). This is the only surface addons may rely on;
    it is kept stable across GUI releases.
    """

    #: API version, as a tuple. Bumped on breaking changes.
    version = API_VERSION

    def __init__(self, manager: "AddonManager", addon: Addon):
        self._manager = manager
        self._addon = addon

    @property
    def gui(self):
        """
        The :class:`SionnaRtGui` instance. Escape hatch, explicitly UNSTABLE:
        anything accessed through it may break without notice in any release.
        """
        return self._manager.gui

    def add_panel(self, title: str, draw_fn: Callable) -> None:
        """
        Add an ImGui panel drawn every frame. ``draw_fn(psim)`` receives the
        ``polyscope.imgui`` module and must only issue ImGui calls (no blocking
        work — use :meth:`run_async` for that). Panels are removed automatically
        when the addon is disabled.
        """
        self._manager.add_panel(self._addon, title, draw_fn)

    def load_scene(self, path: str) -> None:
        """
        Request loading a Sionna RT scene (XML file path). The load is deferred
        to the start of the next frame, so this is safe to call from
        :meth:`run_async` completion callbacks.
        """
        self._manager.gui.load_scene_requested = str(path)

    def get_scene(self):
        """The currently loaded ``sionna.rt.Scene`` (may be None)."""
        return self._manager.gui.scene

    def notify(self, message: str) -> None:
        """Show a transient in-app notification. Safe to call from any thread."""
        self._manager.notify(f"[{self._addon.name}] {message}")

    def run_async(self, fn: Callable, on_done: Callable | None = None) -> None:
        """
        Run ``fn()`` in a background thread; never block the frame loop with
        long work (network requests, heavy computation). If provided,
        ``on_done(result)`` is called on the main thread with ``fn``'s return
        value. ``fn`` must not touch the GUI or Polyscope; do that in
        ``on_done``. If ``fn`` raises, the user is notified and ``on_done`` is
        not called.
        """
        self._manager.run_async(self._addon, fn, on_done)


class AddonManager:
    """Discovers, validates, and runs addons. One instance per application."""

    def __init__(self, gui):
        self.gui = gui
        self.addons: dict[str, Addon] = {}
        # Panels registered by enabled addons: (addon, title, draw_fn)
        self._panels: list[tuple[Addon, str, Callable]] = []
        # Results from run_async worker threads, drained on the main thread.
        self._async_results: queue.SimpleQueue = queue.SimpleQueue()
        self._notifications: list[tuple[str, float]] = []
        self._notifications_lock = threading.Lock()
        self._install_zip_path: str = ""
        # Floating manager window, toggled e.g. from a toolbar button.
        self.show_manager_window: bool = False
        self._uninstall_target: str | None = None
        # Disables/uninstalls requested from the GUI are deferred to the start
        # of the next tick: an addon's unregister() may free GPU resources
        # (textures, buffers) that the current ImGui frame already references,
        # and freeing them mid-frame crashes the renderer.
        self._pending_actions: list[tuple[str, str]] = []

        self._load_persisted_state()
        self.refresh()

    # --- Discovery

    def refresh(self) -> None:
        """(Re-)discover addons from the user directory and pip entry points."""
        discovered: dict[str, Addon] = {}

        if os.path.isdir(USER_ADDONS_DIR):
            for entry in sorted(os.listdir(USER_ADDONS_DIR)):
                path = os.path.join(USER_ADDONS_DIR, entry)
                if os.path.isfile(os.path.join(path, "__init__.py")):
                    discovered[entry] = Addon(name=entry, source="user", path=path)

        try:
            entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as e:
            entry_points = []
            print(f"[!] Failed to query addon entry points: {e}", file=sys.stderr)
        for ep in entry_points:
            # The user directory takes precedence on name collisions.
            if ep.name not in discovered:
                discovered[ep.name] = Addon(name=ep.name, source="pip", entry_point=ep)

        # Keep state of already-known addons, drop the ones that disappeared.
        for name, addon in list(self.addons.items()):
            if name not in discovered:
                if addon.enabled:
                    self.set_enabled(name, False, persist=False)
                del self.addons[name]
        for name, addon in discovered.items():
            if name not in self.addons:
                self.addons[name] = addon
                # Newly discovered addons are enabled unless the config says
                # otherwise.
                if self._cfg_enabled().get(name, True):
                    self.set_enabled(name, True, persist=False)

    def _import(self, addon: Addon) -> None:
        if addon.source == "user":
            module_name = f"sionna_rt_gui_user_addon_{addon.name}"
            if module_name in sys.modules:
                addon.module = sys.modules[module_name]
                return
            spec = importlib.util.spec_from_file_location(
                module_name,
                os.path.join(addon.path, "__init__.py"),
                submodule_search_locations=[addon.path],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[module_name]
                raise
            addon.module = module
        else:
            addon.module = addon.entry_point.load()
            if addon.path is None:
                module_file = getattr(addon.module, "__file__", None)
                if module_file:
                    addon.path = os.path.dirname(os.path.realpath(module_file))

    def _validate(self, addon: Addon) -> str | None:
        """Returns an error message, or None if the addon is valid."""
        info = getattr(addon.module, "ADDON_INFO", None)
        if not isinstance(info, dict):
            return "ADDON_INFO dict is missing"
        if not info.get("name"):
            return 'ADDON_INFO is missing a "name"'
        if not callable(getattr(addon.module, "register", None)):
            return "register(api) function is missing"

        min_gui_version = info.get("min_gui_version")
        if min_gui_version is not None:
            try:
                required = parse_version(min_gui_version)
            except (TypeError, ValueError):
                return f"invalid min_gui_version: {min_gui_version!r}"
            if tuple(GUI_VERSION) < required:
                current = ".".join(str(v) for v in GUI_VERSION)
                return (
                    f"requires GUI version >= {min_gui_version}, "
                    f"but this is version {current}"
                )

        addon.info = info
        return None

    # --- Enable / disable lifecycle

    def set_enabled(self, name: str, enabled: bool, persist: bool = True) -> None:
        addon = self.addons[name]

        if enabled and not addon.enabled:
            addon.error = None
            try:
                if addon.module is None:
                    self._import(addon)
                addon.error = self._validate(addon)
                if addon.error is None:
                    addon.api = AddonAPI(self, addon)
                    addon.module.register(addon.api)
                    addon.enabled = True
            except Exception as e:
                traceback.print_exc()
                addon.error = f"{type(e).__name__}: {e}"
            if addon.error is not None:
                addon.api = None
                self.notify(f"Failed to enable addon '{name}': {addon.error}")
        elif not enabled and addon.enabled:
            unregister = getattr(addon.module, "unregister", None)
            if callable(unregister):
                try:
                    unregister(addon.api)
                except Exception as e:
                    traceback.print_exc()
                    addon.error = f"unregister failed: {type(e).__name__}: {e}"
            # Even if the addon's unregister() is missing or broken, make sure
            # all of its UI is removed.
            self._panels = [p for p in self._panels if p[0] is not addon]
            addon.api = None
            addon.enabled = False

        if persist:
            self._cfg_enabled()[name] = addon.enabled
            self._save_persisted_state()

    def _auto_disable(self, addon: Addon, context: str, exc: Exception) -> None:
        traceback.print_exc()
        self.set_enabled(addon.name, False, persist=False)
        addon.error = f"{context}: {type(exc).__name__}: {exc}"
        self.notify(
            f"Addon '{addon.name}' was disabled after an error in {context}: {exc}"
        )

    def reload_addon(self, name: str) -> None:
        addon = self.addons[name]
        if addon.module is None or addon.path is None:
            return
        was_enabled = addon.enabled
        if was_enabled:
            self.set_enabled(name, False, persist=False)
        if addon.source == "user":
            # Modules loaded from a file location are not importable by name,
            # so a re-import from scratch is the reload.
            sys.modules.pop(addon.module.__name__, None)
            addon.module = None
        else:
            try:
                reload_module_recursive(addon.module, allowed_root=addon.path)
            except Exception as e:
                traceback.print_exc()
                addon.error = f"reload failed: {type(e).__name__}: {e}"
                self.notify(f"Failed to reload addon '{name}': {e}")
                return
        if was_enabled:
            self.set_enabled(name, True, persist=False)
            if addon.enabled:
                self.notify(f"Addon '{name}' reloaded.")

    # --- Persistence (enabled / disabled choices)

    def _cfg_enabled(self) -> dict:
        return self.gui.cfg.addons.enabled

    def _load_persisted_state(self) -> None:
        if not os.path.isfile(ADDONS_STATE_PATH):
            return
        try:
            state = OmegaConf.to_object(OmegaConf.load(ADDONS_STATE_PATH))
        except Exception as e:
            print(
                f"[!] Failed to load addons state from {ADDONS_STATE_PATH}: {e}",
                file=sys.stderr,
            )
            return
        for name, enabled in ((state or {}).get("enabled") or {}).items():
            # Values set explicitly in the main config file take precedence.
            self._cfg_enabled().setdefault(name, bool(enabled))

    def _save_persisted_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(ADDONS_STATE_PATH), exist_ok=True)
            OmegaConf.save(
                OmegaConf.create({"enabled": dict(self._cfg_enabled())}),
                ADDONS_STATE_PATH,
            )
        except Exception as e:
            print(
                f"[!] Failed to save addons state to {ADDONS_STATE_PATH}: {e}",
                file=sys.stderr,
            )

    # --- Install from zip

    def install_from_zip(self, zip_path: str) -> None:
        zip_path = os.path.expanduser(zip_path.strip())
        try:
            name = self._extract_zip(zip_path)
        except Exception as e:
            traceback.print_exc()
            self.notify(f"Failed to install addon from '{zip_path}': {e}")
            return
        self.notify(f"Installed addon '{name}' to {USER_ADDONS_DIR}.")
        self.refresh()

    def _extract_zip(self, zip_path: str) -> str:
        """Extract an addon zip into the user addons directory; returns its name."""
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.strip("/")]
            for n in names:
                parts = n.replace("\\", "/").split("/")
                if n.startswith(("/", "\\")) or ".." in parts or (":" in parts[0]):
                    raise ValueError(f"unsafe path in zip: {n!r}")

            top_level = {n.replace("\\", "/").split("/")[0] for n in names}
            os.makedirs(USER_ADDONS_DIR, exist_ok=True)
            if len(top_level) == 1 and f"{next(iter(top_level))}/__init__.py" in names:
                # Zip contains a single addon directory: extract it as-is.
                name = next(iter(top_level))
                zf.extractall(USER_ADDONS_DIR)
            elif "__init__.py" in names:
                # Flat zip: extract into a directory named after the zip file.
                name = re.sub(
                    r"[^A-Za-z0-9_\-]", "_",
                    os.path.splitext(os.path.basename(zip_path))[0],
                )
                zf.extractall(os.path.join(USER_ADDONS_DIR, name))
            else:
                raise ValueError("no addon __init__.py found in the zip")
        return name

    # --- Async work & notifications

    def run_async(
        self, addon: Addon, fn: Callable, on_done: Callable | None
    ) -> None:
        def target():
            try:
                result = fn()
                error = None
            except Exception as e:
                traceback.print_exc()
                result, error = None, e
            self._async_results.put((addon, on_done, result, error))

        threading.Thread(
            target=target, daemon=True, name=f"addon-{addon.name}"
        ).start()

    def notify(self, message: str) -> None:
        print(f"[i] {message}")
        with self._notifications_lock:
            self._notifications.append((message, time.time()))

    # --- Per-frame work

    def tick(self) -> None:
        """Called once per frame from the application's tick()."""
        # Deferred disables/uninstalls, before anything is drawn this frame.
        for action, name in self._pending_actions:
            if name not in self.addons:
                continue
            if action == "disable":
                self.set_enabled(name, False)
            elif action == "disable_session":
                # Closing a panel window only disables for this session; the
                # addon comes back on the next start.
                self.set_enabled(name, False, persist=False)
            elif action == "uninstall":
                self.uninstall(name)
        self._pending_actions.clear()

        # Deliver completed background tasks on the main thread.
        while True:
            try:
                addon, on_done, result, error = self._async_results.get_nowait()
            except queue.Empty:
                break
            if error is not None:
                self.notify(f"[{addon.name}] Background task failed: {error}")
            elif on_done is not None and addon.enabled:
                try:
                    on_done(result)
                except Exception as e:
                    self._auto_disable(addon, "run_async on_done callback", e)

        if self.gui.cfg.gui_mode != GuiMode.HIDDEN:
            self._draw_panels()
            if self.show_manager_window:
                self._draw_manager_window()
            self._draw_notifications()

    def _draw_panels(self) -> None:
        ui_scale = self.gui.ui_scale
        for i, (addon, title, draw_fn) in enumerate(list(self._panels)):
            psim.SetNextWindowPos(
                ((450 + 30 * i) * ui_scale, (10 + 30 * i) * ui_scale),
                psim.ImGuiCond_FirstUseEver,
            )
            psim.SetNextWindowSize(
                (350 * ui_scale, 250 * ui_scale), psim.ImGuiCond_FirstUseEver
            )
            _, keep_open = psim.Begin(f"{title}##addon_{addon.name}_{i}", open=True)
            try:
                draw_fn(psim)
            except Exception as e:
                self._auto_disable(addon, f"panel '{title}'", e)
            finally:
                psim.End()
            if not keep_open:
                # Closing the panel window disables the addon (deferred to the
                # next frame, see _pending_actions).
                self._pending_actions.append(("disable_session", addon.name))
                self.notify(
                    f"Addon '{addon.name}' disabled. It can be re-enabled in "
                    'the "Addons" section.'
                )

    def add_panel(self, addon: Addon, title: str, draw_fn: Callable) -> None:
        self._panels.append((addon, title, draw_fn))

    def _draw_notifications(self) -> None:
        now = time.time()
        with self._notifications_lock:
            self._notifications = [
                n for n in self._notifications if now - n[1] < NOTIFICATION_DURATION_S
            ]
            messages = [n[0] for n in self._notifications]
        if not messages:
            return

        ui_scale = self.gui.ui_scale
        window_resolution = ps.get_window_size()
        height = (len(messages) * 22 + 16) * ui_scale
        psim.SetNextWindowPos(
            (10 * ui_scale, window_resolution[1] - height - 10 * ui_scale)
        )
        psim.SetNextWindowSize((500 * ui_scale, height))
        psim.Begin(
            "##addon_notifications",
            open=True,
            flags=(
                psim.ImGuiWindowFlags_NoTitleBar
                | psim.ImGuiWindowFlags_NoDecoration
                | psim.ImGuiWindowFlags_NoFocusOnAppearing
                | psim.ImGuiWindowFlags_NoNav
            ),
        )
        for message in messages:
            psim.Text(message)
        psim.End()

    # --- Uninstall (user addons only)

    def uninstall(self, name: str) -> None:
        addon = self.addons.get(name)
        if addon is None or addon.source != "user" or addon.path is None:
            return
        path = os.path.realpath(addon.path)
        if os.path.dirname(path) != os.path.realpath(USER_ADDONS_DIR):
            self.notify(f"Refusing to uninstall '{name}': unexpected path {path}")
            return
        self.set_enabled(name, False, persist=False)
        if addon.module is not None:
            sys.modules.pop(addon.module.__name__, None)
        try:
            shutil.rmtree(path)
        except OSError as e:
            traceback.print_exc()
            self.notify(f"Failed to uninstall addon '{name}': {e}")
            return
        self._cfg_enabled().pop(name, None)
        self._save_persisted_state()
        self.refresh()
        self.notify(f"Uninstalled addon '{name}'.")

    # --- Manager GUI

    def _draw_manager_window(self) -> None:
        ui_scale = self.gui.ui_scale
        window_resolution = ps.get_window_size()
        psim.SetNextWindowSize(
            (540 * ui_scale, 380 * ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (0.5 * (window_resolution[0] - 540 * ui_scale), 60 * ui_scale),
            psim.ImGuiCond_FirstUseEver,
        )
        _, self.show_manager_window = psim.Begin("Addons##addon_manager", open=True)
        try:
            self.draw_manager_gui()
        finally:
            psim.End()

    def draw_manager_gui(self) -> None:
        psim.Spacing()

        if psim.Button("Refresh##addons"):
            self.refresh()
        psim.SameLine()
        psim.Text(f"{len(self.addons)} addon(s) found")

        for name in sorted(self.addons):
            addon = self.addons[name]
            changed, enabled = psim.Checkbox(
                f"{addon.display_name} ({addon.source})##addon_enable_{name}",
                addon.enabled,
            )
            if changed:
                if enabled:
                    self.set_enabled(name, True)
                else:
                    self._pending_actions.append(("disable", name))

            if addon.path is not None and addon.module is not None:
                psim.SameLine()
                if psim.Button(f"Reload##addon_reload_{name}"):
                    self.reload_addon(name)

            if addon.source == "user":
                psim.SameLine()
                if psim.Button(f"Uninstall##addon_uninstall_{name}"):
                    self._uninstall_target = name
                    psim.OpenPopup("Uninstall addon?##addons")

            description = addon.info.get("description")
            if description:
                psim.PushStyleColor(psim.ImGuiCol_Text, (0.6, 0.6, 0.6, 1.0))
                psim.Text(f"    {description}")
                psim.PopStyleColor()
            if addon.error is not None:
                psim.PushStyleColor(psim.ImGuiCol_Text, (1.0, 0.4, 0.4, 1.0))
                psim.Text(f"    Error: {addon.error}")
                psim.PopStyleColor()

        psim.Spacing()
        psim.SeparatorText("Install from zip")
        _, self._install_zip_path = psim.InputText(
            "##addon_zip_path", self._install_zip_path
        )
        psim.SameLine()
        if psim.Button("Install##addon_install"):
            self.install_from_zip(self._install_zip_path)
        psim.PushStyleColor(psim.ImGuiCol_Text, (0.6, 0.6, 0.6, 1.0))
        psim.Text(f"Addons are installed to {USER_ADDONS_DIR}")
        psim.PopStyleColor()

        # Uninstall confirmation dialog
        result = psim.BeginPopupModal(
            "Uninstall addon?##addons",
            True,
            psim.ImGuiWindowFlags_AlwaysAutoResize,
        )
        if result[0] if isinstance(result, tuple) else result:
            psim.Text(
                f"Permanently delete addon '{self._uninstall_target}' from disk?"
            )
            psim.Spacing()
            if psim.Button("Uninstall##addon_uninstall_confirm"):
                self._pending_actions.append(("uninstall", self._uninstall_target))
                self._uninstall_target = None
                psim.CloseCurrentPopup()
            psim.SameLine()
            if psim.Button("Cancel##addon_uninstall_cancel"):
                self._uninstall_target = None
                psim.CloseCurrentPopup()
            psim.EndPopup()

        psim.Spacing()
