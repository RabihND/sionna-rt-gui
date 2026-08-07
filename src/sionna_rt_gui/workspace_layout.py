#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Docked workspace layout: non-overlapping areas with draggable borders, a
header per area, and split property rows (right-aligned label column with the
widget filling the rest of the row).

ImGui is built here without docking support, so areas are laid out by hand:
every frame each area is pinned to a computed rectangle, and thin invisible
buttons between them act as splitters.
"""
from __future__ import annotations

from dataclasses import dataclass

import polyscope as ps
import polyscope.imgui as psim

# --- Dark editor palette
EDITOR_BG = (0.188, 0.188, 0.188, 1.0)  # #303030 editor background
REGION_BG = (0.114, 0.114, 0.114, 1.0)  # #1d1d1d region / panel background
HEADER_BG = (0.137, 0.137, 0.137, 1.0)  # #232323 area headers
PANEL_BG = (0.220, 0.220, 0.220, 1.0)  # #383838 sub-panel background
WIDGET_BG = (0.329, 0.329, 0.329, 1.0)  # #545454 widget inner
WIDGET_HOVER = (0.392, 0.392, 0.392, 1.0)
TEXT = (0.898, 0.898, 0.898, 1.0)  # #e5e5e5
TEXT_DIM = (0.588, 0.588, 0.588, 1.0)
SELECT_BLUE = (0.278, 0.447, 0.702, 1.0)  # #4772b3 selection / active
ACTIVE_ORANGE = (0.929, 0.620, 0.361, 1.0)  # #ed9e5c active object
BORDER = (0.098, 0.098, 0.098, 1.0)

# Area border thickness, in unscaled pixels
SPLITTER = 4.0


def set_editor_imgui_style() -> None:
    """Apply the dark editor theme to ImGui."""

    def style_cb():
        style = psim.GetStyle()

        style.WindowRounding = 0.0
        style.ChildRounding = 4.0
        style.FrameRounding = 4.0
        style.GrabRounding = 4.0
        style.TabRounding = 4.0
        style.PopupRounding = 4.0
        style.ScrollbarRounding = 6.0
        style.WindowPadding = (8, 6)
        style.FramePadding = (7, 4)
        style.ItemSpacing = (6, 5)
        style.ItemInnerSpacing = (5, 4)
        style.CellPadding = (6, 3)
        style.IndentSpacing = 16.0
        style.GrabMinSize = 10.0
        style.ScrollbarSize = 11.0
        style.WindowBorderSize = 1.0
        style.ChildBorderSize = 1.0
        # A hairline outline around every widget: this is what separates a
        # designed panel from a stack of flat rectangles.
        style.FrameBorderSize = 1.0
        style.PopupBorderSize = 1.0
        style.TabBorderSize = 0.0
        style.WindowTitleAlign = (0.0, 0.5)
        style.SeparatorTextBorderSize = 1.0

        style.ScaleAllSizes(ps.get_ui_scale())

        colors = style.Colors
        colors[psim.ImGuiCol_WindowBg] = REGION_BG
        colors[psim.ImGuiCol_ChildBg] = PANEL_BG
        colors[psim.ImGuiCol_PopupBg] = (0.157, 0.157, 0.157, 0.98)
        colors[psim.ImGuiCol_MenuBarBg] = HEADER_BG
        colors[psim.ImGuiCol_Border] = BORDER
        colors[psim.ImGuiCol_BorderShadow] = (0.0, 0.0, 0.0, 0.0)

        colors[psim.ImGuiCol_Text] = TEXT
        colors[psim.ImGuiCol_TextDisabled] = TEXT_DIM
        colors[psim.ImGuiCol_TextSelectedBg] = SELECT_BLUE

        # Flat mid-grey widgets, lighter on hover
        colors[psim.ImGuiCol_FrameBg] = WIDGET_BG
        colors[psim.ImGuiCol_FrameBgHovered] = WIDGET_HOVER
        colors[psim.ImGuiCol_FrameBgActive] = (0.278, 0.353, 0.478, 1.0)
        colors[psim.ImGuiCol_Button] = (0.345, 0.345, 0.345, 1.0)
        colors[psim.ImGuiCol_ButtonHovered] = (0.416, 0.416, 0.416, 1.0)
        colors[psim.ImGuiCol_ButtonActive] = SELECT_BLUE

        # Collapsible sub-panel headers
        colors[psim.ImGuiCol_Header] = (0.239, 0.239, 0.239, 1.0)
        colors[psim.ImGuiCol_HeaderHovered] = (0.290, 0.290, 0.290, 1.0)
        colors[psim.ImGuiCol_HeaderActive] = (0.263, 0.263, 0.263, 1.0)

        colors[psim.ImGuiCol_TitleBg] = HEADER_BG
        colors[psim.ImGuiCol_TitleBgActive] = HEADER_BG
        colors[psim.ImGuiCol_TitleBgCollapsed] = HEADER_BG

        colors[psim.ImGuiCol_Tab] = (0.157, 0.157, 0.157, 1.0)
        colors[psim.ImGuiCol_TabHovered] = (0.263, 0.263, 0.263, 1.0)
        colors[psim.ImGuiCol_TabSelected] = PANEL_BG
        colors[psim.ImGuiCol_TabDimmed] = (0.137, 0.137, 0.137, 1.0)
        colors[psim.ImGuiCol_TabDimmedSelected] = (0.196, 0.196, 0.196, 1.0)

        colors[psim.ImGuiCol_CheckMark] = (0.831, 0.831, 0.831, 1.0)
        colors[psim.ImGuiCol_SliderGrab] = (0.451, 0.451, 0.451, 1.0)
        colors[psim.ImGuiCol_SliderGrabActive] = SELECT_BLUE
        colors[psim.ImGuiCol_ResizeGrip] = (0.0, 0.0, 0.0, 0.0)
        colors[psim.ImGuiCol_ResizeGripHovered] = SELECT_BLUE
        colors[psim.ImGuiCol_ResizeGripActive] = SELECT_BLUE
        colors[psim.ImGuiCol_NavHighlight] = SELECT_BLUE
        colors[psim.ImGuiCol_DragDropTarget] = ACTIVE_ORANGE

        colors[psim.ImGuiCol_Separator] = BORDER
        colors[psim.ImGuiCol_SeparatorHovered] = SELECT_BLUE
        colors[psim.ImGuiCol_SeparatorActive] = SELECT_BLUE

        colors[psim.ImGuiCol_ScrollbarBg] = (0.098, 0.098, 0.098, 1.0)
        colors[psim.ImGuiCol_ScrollbarGrab] = (0.318, 0.318, 0.318, 1.0)
        colors[psim.ImGuiCol_ScrollbarGrabHovered] = (0.392, 0.392, 0.392, 1.0)
        colors[psim.ImGuiCol_ScrollbarGrabActive] = (0.451, 0.451, 0.451, 1.0)
        colors[psim.ImGuiCol_PlotHistogram] = SELECT_BLUE

    ps.set_configure_imgui_style_callback(style_cb)


Rect = tuple[float, float, float, float]  # x, y, width, height


@dataclass
class AreaLayout:
    """
    Sizes of the fixed areas, in unscaled pixels (or fractions where noted).
    These are what the splitters edit, and they persist for the session.
    """

    topbar_height: float = 32.0
    status_height: float = 26.0
    tools_width: float = 132.0
    side_width: float = 340.0
    # Share of the side column taken by the outliner (rest goes to properties)
    outliner_fraction: float = 0.34
    timeline_height: float = 118.0

    def clamp(self, window_width: float, window_height: float, scale: float) -> None:
        self.tools_width = float(
            min(max(self.tools_width, 0.0), 0.3 * window_width / scale)
        )
        self.side_width = float(
            min(max(self.side_width, 180.0), 0.6 * window_width / scale)
        )
        self.timeline_height = float(
            min(max(self.timeline_height, 0.0), 0.5 * window_height / scale)
        )
        self.outliner_fraction = float(min(max(self.outliner_fraction, 0.1), 0.85))

    def areas(self, window_size: tuple[int, int], scale: float) -> dict[str, Rect]:
        width, height = float(window_size[0]), float(window_size[1])
        self.clamp(width, height, scale)

        topbar_h = self.topbar_height * scale
        status_h = self.status_height * scale
        tools_w = self.tools_width * scale
        side_w = self.side_width * scale
        timeline_h = self.timeline_height * scale

        middle_y = topbar_h
        middle_h = height - topbar_h - status_h
        side_x = width - side_w
        outliner_h = self.outliner_fraction * middle_h

        center_x = tools_w
        center_w = side_x - tools_w
        return {
            "topbar": (0.0, 0.0, width, topbar_h),
            "tools": (0.0, middle_y, tools_w, middle_h),
            # The 3D view is the uncovered region: reported for the camera,
            # never drawn over.
            "viewport": (center_x, middle_y, center_w, middle_h - timeline_h),
            "timeline": (center_x, middle_y + middle_h - timeline_h, center_w, timeline_h),
            "outliner": (side_x, middle_y, side_w, outliner_h),
            "properties": (side_x, middle_y + outliner_h, side_w, middle_h - outliner_h),
            "status": (0.0, height - status_h, width, status_h),
        }


AREA_FLAGS = (
    psim.ImGuiWindowFlags_NoTitleBar
    | psim.ImGuiWindowFlags_NoMove
    | psim.ImGuiWindowFlags_NoResize
    | psim.ImGuiWindowFlags_NoCollapse
    | psim.ImGuiWindowFlags_NoBringToFrontOnFocus
    | psim.ImGuiWindowFlags_NoSavedSettings
)


def begin_area(
    name: str, rect: Rect, extra_flags: int = 0, background=None
) -> bool:
    """Pin a window to the given rectangle as a fixed area."""
    psim.SetNextWindowPos((rect[0], rect[1]))
    psim.SetNextWindowSize((rect[2], rect[3]))
    if background is not None:
        psim.PushStyleColor(psim.ImGuiCol_WindowBg, background)
    opened = psim.Begin(name, open=True, flags=AREA_FLAGS | extra_flags)
    if background is not None:
        psim.PopStyleColor()
    return opened


def end_area() -> None:
    psim.End()


def area_header(title: str, scale: float) -> None:
    """Small darker strip carrying the area's name."""
    draw_list = psim.GetWindowDrawList()
    x, y = psim.GetCursorScreenPos()
    width = psim.GetContentRegionAvail()[0]
    height = psim.GetFrameHeight()
    padding = psim.GetStyle().WindowPadding
    draw_list.AddRectFilled(
        (x - padding[0], y - padding[1]),
        (x + width + padding[0], y + height),
        _packed(HEADER_BG),
    )
    psim.SetCursorPosY(psim.GetCursorPosY() + 2 * scale)
    psim.TextDisabled(title.upper())
    psim.Dummy((0.0, 3 * scale))
    psim.Separator()
    psim.Dummy((0.0, 2 * scale))


def _packed(color) -> int:
    r, g, b, a = color
    return (
        (int(a * 255) << 24) | (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)
    )


def splitter(
    identifier: str, rect: Rect, vertical: bool, scale: float
) -> float:
    """
    Draggable border between two areas. Returns the drag delta in unscaled
    pixels (positive to the right / downwards), zero when not dragged.
    """
    thickness = SPLITTER * scale
    if vertical:
        area = (rect[0] - 0.5 * thickness, rect[1], thickness, rect[3])
    else:
        area = (rect[0], rect[1] - 0.5 * thickness, rect[2], thickness)

    psim.SetNextWindowPos((area[0], area[1]))
    psim.SetNextWindowSize((area[2], area[3]))
    psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (0.0, 0.0))
    psim.Begin(
        f"##splitter_{identifier}",
        open=True,
        flags=AREA_FLAGS
        | psim.ImGuiWindowFlags_NoBackground
        | psim.ImGuiWindowFlags_NoScrollbar,
    )
    psim.InvisibleButton(f"##grip_{identifier}", (max(area[2], 1.0), max(area[3], 1.0)))
    hovered = psim.IsItemHovered()
    active = psim.IsItemActive()
    if hovered or active:
        psim.SetMouseCursor(
            psim.ImGuiMouseCursor_ResizeEW if vertical else psim.ImGuiMouseCursor_ResizeNS
        )
        psim.GetWindowDrawList().AddRectFilled(
            (area[0], area[1]),
            (area[0] + area[2], area[1] + area[3]),
            _packed(SELECT_BLUE),
        )
    delta = 0.0
    if active:
        mouse_delta = psim.GetIO().MouseDelta
        delta = (mouse_delta[0] if vertical else mouse_delta[1]) / scale
    psim.End()
    psim.PopStyleVar()
    return delta


def tab_strip(labels: list[str], current: int, scale: float) -> int:
    """
    Row of tab buttons that wraps to the panel width. The active tab is
    highlighted. Returns the selected index.
    """
    selected = min(max(current, 0), len(labels) - 1)
    style = psim.GetStyle()
    available = psim.GetContentRegionAvail()[0]
    used = 0.0
    for i, label in enumerate(labels):
        width = psim.CalcTextSize(label)[0] + 2 * style.FramePadding[0]
        if i > 0:
            if used + style.ItemSpacing[0] + width <= available:
                psim.SameLine()
                used += style.ItemSpacing[0] + width
            else:
                used = width
        else:
            used = width

        is_active = i == selected
        if is_active:
            psim.PushStyleColor(psim.ImGuiCol_Button, SELECT_BLUE)
            psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, SELECT_BLUE)
        if psim.Button(f"{label}##tab_{i}"):
            selected = i
        if is_active:
            psim.PopStyleColor(2)
    psim.Dummy((0.0, 2 * scale))
    psim.Separator()
    return selected


def property_row(label: str, scale: float, label_fraction: float = 0.42) -> None:
    """
    Start a split property row: the label is right-aligned in the left
    column, and the caller's widget fills the rest of the row. Call once per
    widget, then draw the widget with an empty label:

        property_row("Cell size", scale)
        changed, value = psim.SliderFloat("##cell_size", value, 0.1, 10.0)
    """
    available = psim.GetContentRegionAvail()[0]
    label_width = label_fraction * available
    text_width = psim.CalcTextSize(label)[0]
    psim.AlignTextToFramePadding()
    # Right-align the label against the widget column
    psim.SetCursorPosX(
        psim.GetCursorPosX() + max(label_width - text_width - 8 * scale, 0.0)
    )
    psim.Text(label)
    psim.SameLine()
    psim.SetCursorPosX(psim.GetCursorPosX() + max(0.0, 0.0))
    psim.PushItemWidth(-1.0)


def end_property_row() -> None:
    psim.PopItemWidth()


def numeric_field(
    identifier: str,
    value: float,
    speed: float,
    minimum: float,
    maximum: float,
    fmt: str = "%.2f",
    tooltip: str = "",
) -> tuple[bool, float]:
    """
    A draggable number that can also be typed into. Dragging is quick but
    imprecise, so every one of these advertises that a value can be entered
    exactly, and typed values are clamped to the range.
    """
    changed, new_value = psim.DragFloat(
        identifier,
        value,
        speed,
        minimum,
        maximum,
        fmt,
        psim.ImGuiSliderFlags_AlwaysClamp,
    )
    if psim.IsItemHovered():
        psim.SetTooltip(
            (tooltip + "\n" if tooltip else "")
            + "Drag to adjust, or Ctrl+click to type an exact value"
        )
    return changed, new_value


def numeric_field3(
    identifier: str,
    values,
    speed: float,
    fmt: str = "%.2f",
    tooltip: str = "",
):
    """Three draggable numbers that can also be typed into."""
    changed, new_values = psim.DragFloat3(identifier, tuple(values), speed, 0.0, 0.0, fmt)
    if psim.IsItemHovered():
        psim.SetTooltip(
            (tooltip + "\n" if tooltip else "")
            + "Drag to adjust, or Ctrl+click a number to type it"
        )
    return changed, new_values
