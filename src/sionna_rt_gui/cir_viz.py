#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Channel impulse response viewer: a stem plot of the path coefficients
|a_i| (in dB) over their propagation delays, for a selectable TX-RX pair.
Uses the CIR already computed by Paths.cir() when `compute_cir` is enabled.
"""
from __future__ import annotations

import numpy as np
import polyscope.imgui as psim

from .ps_utils import ACCENT_BRIGHT, im_col32

# Dynamic range shown below the strongest path
CIR_DB_RANGE = 60.0


def _pair_cir(
    gui: "SionnaRtGui", rx_i: int, tx_i: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract |a| and delays [s] for one TX-RX pair (first antenna element of
    each array, first time step), keeping only valid paths.
    """
    a, tau = gui.paths_cir
    amplitude = np.abs(a[rx_i, 0, tx_i, 0, :, 0])
    if tau.ndim == 3:  # synthetic array: [num_rx, num_tx, num_paths]
        delays = tau[rx_i, tx_i, :]
    else:  # per-antenna: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        delays = tau[rx_i, 0, tx_i, 0, :]
    valid = (delays >= 0.0) & (amplitude > 0.0)
    return amplitude[valid], delays[valid]


def _rms_delay_spread(amplitude: np.ndarray, delays: np.ndarray) -> float:
    power = amplitude**2
    total = np.sum(power)
    if total <= 0.0:
        return 0.0
    mean_delay = np.sum(power * delays) / total
    return float(np.sqrt(np.sum(power * (delays - mean_delay) ** 2) / total))


def cir_window(gui: "SionnaRtGui") -> None:
    """
    The channel impulse response in its own floating window, for the classic
    layout. Shown while `compute_cir` is enabled in the Paths section.
    """
    if not gui.cfg.paths.compute_cir:
        return

    s = gui.ui_scale
    psim.SetNextWindowSize((640 * s, 400 * s), psim.ImGuiCond_FirstUseEver)
    psim.SetNextWindowPos((450 * s, 480 * s), psim.ImGuiCond_FirstUseEver)
    _, keep_open = psim.Begin("Channel impulse response###cir_window", open=True)
    if not keep_open:
        gui.cfg.paths.compute_cir = False
        psim.End()
        return
    cir_contents(gui)
    psim.End()


def cir_contents(gui: "SionnaRtGui") -> None:
    """
    Stem plot of the channel impulse response for a selectable TX-RX pair,
    drawn into the current window.
    """
    s = gui.ui_scale
    if not gui.cfg.paths.compute_cir:
        psim.TextDisabled(
            "The channel impulse response is not being computed."
        )
        if psim.Button("Compute it##cir_enable"):
            gui.cfg.paths.compute_cir = True
            gui.update_paths(show=True)
        return

    if gui.paths_cir is None:
        psim.TextDisabled(
            "No paths yet. Add at least one transmitter and one receiver, "
            "then compute paths."
        )
        return

    a, _ = gui.paths_cir
    num_rx, num_tx = a.shape[0], a.shape[2]
    tx_names = list(gui.scene._transmitters.keys())
    rx_names = list(gui.scene._receivers.keys())

    # TX / RX pair selection (only shown when there is a choice)
    pair = getattr(gui, "cir_pair", None) or [0, 0]
    pair[0] = min(pair[0], num_rx - 1)
    pair[1] = min(pair[1], num_tx - 1)
    if num_tx > 1 or num_rx > 1:
        psim.PushItemWidth(160 * s)
        _, pair[1] = psim.Combo("TX##cir", pair[1], tx_names[:num_tx])
        psim.SameLine()
        _, pair[0] = psim.Combo("RX##cir", pair[0], rx_names[:num_rx])
        psim.PopItemWidth()
    gui.cir_pair = pair
    rx_i, tx_i = pair

    amplitude, delays = _pair_cir(gui, rx_i, tx_i)
    if amplitude.size == 0:
        psim.TextDisabled(
            f"No propagation paths between "
            f"{tx_names[tx_i] if tx_i < len(tx_names) else tx_i} and "
            f"{rx_names[rx_i] if rx_i < len(rx_names) else rx_i}."
        )
        return

    power_db = 20.0 * np.log10(amplitude)
    delays_ns = delays * 1e9

    # Delay axis: either following the data, or fixed by the user so that
    # changes stay comparable while devices move. The fixed default comes
    # from the scenario size: the time light needs to cross the scene
    # diagonal (0.3 m per ns).
    if gui.cir_max_delay_ns <= 0.0:
        extents = np.array(gui.scene.mi_scene.bbox().extents())
        diagonal_ns = float(np.linalg.norm(extents)) / 0.3
        gui.cir_max_delay_ns = max(np.ceil(diagonal_ns / 50.0) * 50.0, 50.0)
    _, gui.cir_auto_delay = psim.Checkbox("Auto##cir_xaxis", gui.cir_auto_delay)
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Auto: the delay axis rescales to the current paths.\n"
            "Off: fixed maximum delay, so changes stay comparable."
        )
    psim.SameLine()
    psim.BeginDisabled(gui.cir_auto_delay)
    psim.PushItemWidth(120 * gui.ui_scale)
    _, gui.cir_max_delay_ns = psim.DragFloat(
        "Max delay [ns]##cir_xaxis",
        gui.cir_max_delay_ns,
        1.0,
        10.0,
        100000.0,
        format="%.0f",
    )
    psim.PopItemWidth()
    psim.EndDisabled()

    # Level axis: same idea for the vertical dB scale.
    if gui.cir_max_db is None:
        gui.cir_max_db = float(np.ceil(np.max(power_db) / 5.0) * 5.0)
    _, gui.cir_auto_level = psim.Checkbox("Auto##cir_yaxis", gui.cir_auto_level)
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Auto: the dB axis rescales to the strongest path.\n"
            "Off: fixed maximum level, so changes stay comparable."
        )
    psim.SameLine()
    psim.BeginDisabled(gui.cir_auto_level)
    psim.PushItemWidth(120 * gui.ui_scale)
    _, gui.cir_max_db = psim.DragFloat(
        "Max level [dB]##cir_yaxis",
        gui.cir_max_db,
        1.0,
        -300.0,
        100.0,
        format="%.0f",
    )
    psim.PopItemWidth()
    psim.EndDisabled()

    if gui.cir_auto_delay:
        delay_max = max(float(np.max(delays_ns)) * 1.08, 1.0)
    else:
        delay_max = max(gui.cir_max_delay_ns, 10.0)
    if gui.cir_auto_level:
        db_max = float(np.ceil(np.max(power_db) / 5.0) * 5.0)
    else:
        db_max = gui.cir_max_db
    db_floor = db_max - CIR_DB_RANGE

    # Stats line
    strongest = float(np.max(power_db))
    spread_ns = _rms_delay_spread(amplitude, delays) * 1e9
    psim.Text(
        f"{amplitude.size} path{'s' if amplitude.size != 1 else ''} | "
        f"strongest {strongest:.1f} dB | "
        f"RMS delay spread {spread_ns:.1f} ns"
    )
    psim.TextDisabled(
        "|h| per path [dB], first antenna element. Delays are relative "
        "to the first arrival."
    )

    # --- Plot area
    draw_list = psim.GetWindowDrawList()
    origin = psim.GetCursorScreenPos()
    avail = psim.GetContentRegionAvail()
    margin_left = 42 * s
    margin_bottom = 24 * s
    plot_w = max(avail[0] - margin_left - 8 * s, 60 * s)
    plot_h = max(avail[1] - margin_bottom - 6 * s, 60 * s)
    x0 = origin[0] + margin_left
    y0 = origin[1]

    text_col = im_col32(0.62, 0.64, 0.65, 0.9)
    grid_col = im_col32(0.55, 0.57, 0.58, 0.22)
    axis_col = im_col32(0.55, 0.57, 0.58, 0.55)

    def to_screen(delay_ns: float, db: float) -> tuple[float, float]:
        fx = delay_ns / delay_max
        fy = (db_max - db) / (db_max - db_floor)
        return x0 + fx * plot_w, y0 + np.clip(fy, 0.0, 1.0) * plot_h

    # Horizontal grid every 10 dB
    db_line = db_max
    while db_line >= db_floor:
        px, py = to_screen(0.0, db_line)
        draw_list.AddLine((px, py), (px + plot_w, py), grid_col, 1.0)
        draw_list.AddText((origin[0], py - 7 * s), text_col, f"{db_line:.0f}")
        db_line -= 10.0
    # Vertical grid: ~6 delay ticks
    tick = max(round(delay_max / 6 / 5) * 5, 1)
    delay_line = 0.0
    while delay_line <= delay_max:
        px, py = to_screen(delay_line, db_floor)
        draw_list.AddLine((px, y0), (px, py), grid_col, 1.0)
        draw_list.AddText(
            (px - 8 * s, py + 4 * s), text_col, f"{delay_line:.0f}"
        )
        delay_line += tick
    draw_list.AddText(
        (x0 + plot_w - 60 * s, y0 + plot_h + 4 * s), text_col, "delay [ns]"
    )
    # Axes
    draw_list.AddLine((x0, y0), (x0, y0 + plot_h), axis_col, 1.0)
    draw_list.AddLine(
        (x0, y0 + plot_h), (x0 + plot_w, y0 + plot_h), axis_col, 1.0
    )

    # Stems
    stem_col = im_col32(*ACCENT_BRIGHT)
    for delay_ns, db in zip(delays_ns, power_db):
        if db < db_floor or delay_ns > delay_max:
            continue
        px, py = to_screen(delay_ns, db)
        draw_list.AddLine((px, y0 + plot_h), (px, py), stem_col, 1.5 * s)
        draw_list.AddCircleFilled((px, py), 3.0 * s, stem_col)

    psim.Dummy((avail[0], plot_h + margin_bottom))