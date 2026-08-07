#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
import mitsuba as mi
import polyscope as ps
import polyscope.imgui as psim

from .dlpack_utils import pointer_from_dlpack


def get_array_ptr(arr: dr.ArrayBase) -> tuple[int, int, int, int]:
    value = arr.array if dr.is_tensor_v(arr) else arr
    if hasattr(value, "data_"):
        ptr = value.data_()
    else:
        ptr = pointer_from_dlpack(value)

    return (
        # arr_ptr
        ptr,
        # arr_shape
        dr.shape(arr),
        # arr_dtype
        None,
        # arr_nbytes
        get_array_size_bytes(arr),
    )


def get_array_size_bytes(arr: dr.ArrayBase) -> int:
    assert dr.is_tensor_v(arr) or dr.depth_v(arr) == 1
    match dr.type_v(arr):
        case dr.VarType.UInt8:
            ts = 1
        case dr.VarType.Float16:
            ts = 2
        case dr.VarType.Float32:
            ts = 4
        case dr.VarType.Float64:
            ts = 8
        case _:
            raise ValueError(f"Unsupported array type: {dr.type_v(arr)}")
    return ts * dr.width(arr)


def memcpy_2d_to_array_async(dst_ptr, src_ptr, width, height):
    return dr.cuda.memcpy_2d_to_array_async(
        dst_ptr, src_ptr, src_pitch=width, height=height, from_host=False
    )


def set_polyscope_device_interop_funcs():
    ps_device_func_dict = {
        "map_resource_and_get_array": lambda handle: dr.cuda.map_graphics_resource_array(
            handle
        ),
        "map_resource_and_get_pointer": lambda handle: dr.cuda.map_graphics_resource_ptr(
            handle
        ),
        "unmap_resource": lambda handle: dr.cuda.unmap_graphics_resource(handle),
        "register_gl_buffer": lambda native_id: dr.cuda.register_gl_buffer(native_id),
        "register_gl_image_2d": lambda native_id: dr.cuda.register_gl_texture(
            native_id
        ),
        "unregister_resource": lambda handle: dr.cuda.unregister_cuda_resource(handle),
        "get_array_ptr": get_array_ptr,
        "memcpy_2d": memcpy_2d_to_array_async,
    }

    ps.set_device_interop_funcs(ps_device_func_dict)


def supports_direct_update_from_device() -> bool:
    return "cuda" in mi.variant()


# NVIDIA green accent, shared by the theme and custom-drawn UI elements.
ACCENT = (0.46, 0.73, 0.00)
ACCENT_BRIGHT = (0.58, 0.85, 0.12)
ACCENT_DIM = (0.30, 0.46, 0.04)


def im_col32(r: float, g: float, b: float, a: float = 1.0) -> int:
    """Pack a float RGBA color into the packed 32-bit format used by ImDrawList."""
    return (
        (int(a * 255) << 24) | (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)
    )


def set_custom_imgui_style():
    def style_cb():
        """
        Neutral graphite theme with a single NVIDIA-green accent (#76B900),
        reserved for interactive state: checkmarks, slider grabs, active
        headers/tabs and selections. Surfaces stay quiet so the accent and
        the rendered scene carry the color.
        """
        style = psim.GetStyle()

        style.WindowRounding = 6.0
        style.FrameRounding = 4.0
        style.GrabRounding = 3.0
        style.TabRounding = 4.0
        style.PopupRounding = 4.0
        style.ScrollbarRounding = 6.0
        style.WindowPadding = (12, 12)
        style.FramePadding = (8, 5)
        style.ItemSpacing = (8, 7)
        style.ItemInnerSpacing = (6, 4)
        style.IndentSpacing = 18.0
        style.GrabMinSize = 12.0
        style.ScrollbarSize = 12.0
        style.WindowBorderSize = 1.0
        style.FrameBorderSize = 0.0
        style.PopupBorderSize = 1.0
        style.WindowTitleAlign = (0.0, 0.5)

        style.ScaleAllSizes(ps.get_ui_scale())

        # NVIDIA green accent tiers
        accent = ACCENT
        accent_bright = ACCENT_BRIGHT
        accent_dim = ACCENT_DIM

        # Primary background (slightly translucent over the 3D scene)
        style.Colors[psim.ImGuiCol_WindowBg] = (0.094, 0.098, 0.102, 0.97)
        style.Colors[psim.ImGuiCol_ChildBg] = (0.00, 0.00, 0.00, 0.00)
        style.Colors[psim.ImGuiCol_MenuBarBg] = (0.125, 0.13, 0.135, 1.00)

        style.Colors[psim.ImGuiCol_PopupBg] = (0.13, 0.135, 0.14, 0.98)

        # Headers
        style.Colors[psim.ImGuiCol_Header] = (0.165, 0.17, 0.175, 1.00)
        style.Colors[psim.ImGuiCol_HeaderHovered] = (0.215, 0.225, 0.23, 1.00)
        style.Colors[psim.ImGuiCol_HeaderActive] = (*accent_dim, 0.80)

        # Buttons
        style.Colors[psim.ImGuiCol_Button] = (0.20, 0.21, 0.22, 1.00)
        style.Colors[psim.ImGuiCol_ButtonHovered] = (0.265, 0.275, 0.285, 1.00)
        style.Colors[psim.ImGuiCol_ButtonActive] = (*accent_dim, 1.00)

        # Frame BG
        style.Colors[psim.ImGuiCol_FrameBg] = (0.165, 0.17, 0.18, 1.00)
        style.Colors[psim.ImGuiCol_FrameBgHovered] = (0.21, 0.22, 0.23, 1.00)
        style.Colors[psim.ImGuiCol_FrameBgActive] = (0.245, 0.255, 0.265, 1.00)

        # Tabs
        style.Colors[psim.ImGuiCol_Tab] = (0.15, 0.155, 0.16, 1.00)
        style.Colors[psim.ImGuiCol_TabHovered] = (0.24, 0.25, 0.26, 1.00)
        style.Colors[psim.ImGuiCol_TabActive] = (*accent_dim, 1.00)
        style.Colors[psim.ImGuiCol_TabUnfocused] = (0.12, 0.125, 0.13, 1.00)
        style.Colors[psim.ImGuiCol_TabUnfocusedActive] = (0.19, 0.20, 0.205, 1.00)

        # Title bars blend into the window body for a seamless panel look.
        style.Colors[psim.ImGuiCol_TitleBg] = (0.094, 0.098, 0.102, 1.00)
        style.Colors[psim.ImGuiCol_TitleBgActive] = (0.115, 0.125, 0.115, 1.00)
        style.Colors[psim.ImGuiCol_TitleBgCollapsed] = (0.094, 0.098, 0.102, 0.90)

        # Borders
        style.Colors[psim.ImGuiCol_Border] = (0.28, 0.29, 0.30, 0.45)
        style.Colors[psim.ImGuiCol_BorderShadow] = (0.00, 0.00, 0.00, 0.00)

        # Separators
        style.Colors[psim.ImGuiCol_Separator] = (0.26, 0.27, 0.28, 0.60)
        style.Colors[psim.ImGuiCol_SeparatorHovered] = (*accent, 0.60)
        style.Colors[psim.ImGuiCol_SeparatorActive] = (*accent, 1.00)

        # Text
        style.Colors[psim.ImGuiCol_Text] = (0.92, 0.93, 0.93, 1.00)
        style.Colors[psim.ImGuiCol_TextDisabled] = (0.55, 0.57, 0.58, 1.00)
        style.Colors[psim.ImGuiCol_TextSelectedBg] = (*accent, 0.35)

        # Highlights
        style.Colors[psim.ImGuiCol_CheckMark] = (*accent_bright, 1.00)
        style.Colors[psim.ImGuiCol_SliderGrab] = (*accent, 1.00)
        style.Colors[psim.ImGuiCol_SliderGrabActive] = (*accent_bright, 1.00)
        style.Colors[psim.ImGuiCol_ResizeGrip] = (*accent, 0.40)
        style.Colors[psim.ImGuiCol_ResizeGripHovered] = (*accent, 0.70)
        style.Colors[psim.ImGuiCol_ResizeGripActive] = (*accent_bright, 1.00)
        style.Colors[psim.ImGuiCol_NavHighlight] = (*accent, 1.00)
        style.Colors[psim.ImGuiCol_DragDropTarget] = (*accent_bright, 0.90)

        # Scrollbar
        style.Colors[psim.ImGuiCol_ScrollbarBg] = (0.09, 0.095, 0.10, 0.60)
        style.Colors[psim.ImGuiCol_ScrollbarGrab] = (0.28, 0.29, 0.30, 1.00)
        style.Colors[psim.ImGuiCol_ScrollbarGrabHovered] = (0.36, 0.37, 0.38, 1.00)
        style.Colors[psim.ImGuiCol_ScrollbarGrabActive] = (0.44, 0.45, 0.46, 1.00)

    ps.set_configure_imgui_style_callback(style_cb)
