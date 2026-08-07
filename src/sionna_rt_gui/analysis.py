#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Readouts derived from what the solvers already produce: a link budget for a
transmitter-receiver pair, and coverage statistics for a radio map.

Both pull whole arrays back from the device, which is far too expensive to do
every frame, so results are cached and refreshed on a timer or on demand.
"""
from __future__ import annotations

import time

import drjit as dr
import numpy as np
import polyscope.imgui as psim

from .ps_utils import ACCENT_BRIGHT
from .workspace_layout import end_property_row, numeric_field, property_row

# Seconds between automatic refreshes of the cached statistics
REFRESH_INTERVAL_S = 2.0


def _to_db(values: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, floor))


def radio_map_statistics(gui: "SionnaRtGui") -> dict | None:
    """
    Coverage statistics over the radio map, using the best server per cell.
    Returns None when there is no map.
    """
    radio_map = gui.radio_map
    if radio_map is None:
        return None

    # Best server per cell, in watts, and the corresponding SINR
    rss = dr.max(radio_map.rss, axis=0).numpy()
    sinr = dr.max(radio_map.sinr, axis=0).numpy()

    covered = rss > 0.0
    n_cells = int(rss.size)
    n_covered = int(np.count_nonzero(covered))
    if n_covered == 0:
        return {
            "cells": n_cells,
            "covered": 0,
            "samples": gui.rm_accumulated_samples,
        }

    rss_dbm = _to_db(rss[covered]) + 30.0
    sinr_db = _to_db(sinr[covered])
    return {
        "cells": n_cells,
        "covered": n_covered,
        "rss_dbm_mean": float(np.mean(rss_dbm)),
        "rss_dbm_median": float(np.median(rss_dbm)),
        "rss_dbm_max": float(np.max(rss_dbm)),
        "rss_dbm_p10": float(np.percentile(rss_dbm, 10)),
        "sinr_db_mean": float(np.mean(sinr_db)),
        "sinr_db_median": float(np.median(sinr_db)),
        "rss_dbm_all": rss_dbm,
        "samples": gui.rm_accumulated_samples,
    }


def link_budget(gui: "SionnaRtGui", rx_index: int = 0, tx_index: int = 0) -> dict | None:
    """
    Link budget for one transmitter-receiver pair, from the computed paths.
    """
    if gui.paths_cir is None or gui.paths is None:
        return None

    a, tau = gui.paths_cir
    if a.shape[0] <= rx_index or a.shape[2] <= tx_index:
        return None

    amplitude = np.abs(a[rx_index, 0, tx_index, 0, :, 0])
    if tau.ndim == 3:
        delays = tau[rx_index, tx_index, :]
    else:
        delays = tau[rx_index, 0, tx_index, 0, :]
    valid = (delays >= 0.0) & (amplitude > 0.0)
    amplitude, delays = amplitude[valid], delays[valid]
    if amplitude.size == 0:
        return {"paths": 0}

    power = amplitude**2
    total_gain_db = float(_to_db(np.array([np.sum(power)]))[0])
    strongest_db = float(_to_db(np.array([np.max(power)]))[0])

    # Delay spread about the power-weighted mean delay
    mean_delay = float(np.sum(power * delays) / np.sum(power))
    spread = float(np.sqrt(np.sum(power * (delays - mean_delay) ** 2) / np.sum(power)))

    # Rice-like factor: strongest path against the rest
    rest = np.sum(power) - np.max(power)
    k_factor_db = (
        float(_to_db(np.array([np.max(power) / rest]))[0]) if rest > 0 else float("inf")
    )

    transmitters = list(gui.scene._transmitters.values())
    power_dbm = (
        float(transmitters[tx_index].power_dbm[0])
        if tx_index < len(transmitters)
        else 0.0
    )

    received_dbm = power_dbm + total_gain_db
    thermal_dbm = 10.0 * np.log10(float(gui.scene.thermal_noise_power[0])) + 30.0
    # A real receiver adds its own noise on top of the thermal floor
    noise_dbm = thermal_dbm + gui.cfg.noise_figure_db
    result = {
        "thermal_dbm": thermal_dbm,
        "noise_figure_db": gui.cfg.noise_figure_db,
        "paths": int(amplitude.size),
        "path_gain_db": total_gain_db,
        "received_dbm": received_dbm,
        "noise_dbm": noise_dbm,
        "snr_db": received_dbm - noise_dbm,
        "tx_power_dbm": power_dbm,
        "strongest_db": strongest_db,
        "mean_delay_ns": mean_delay * 1e9,
        "delay_spread_ns": spread * 1e9,
        "k_factor_db": k_factor_db,
    }

    # Doppler, when the devices are moving
    try:
        doppler = gui.paths.doppler.numpy().ravel()
        doppler = doppler[np.isfinite(doppler)]
        if doppler.size and np.any(doppler != 0.0):
            result["doppler_max_hz"] = float(np.max(np.abs(doppler)))
            result["doppler_spread_hz"] = float(np.std(doppler))
    except Exception:
        pass
    return result


def refresh_statistics(gui: "SionnaRtGui", force: bool = False) -> None:
    """Recompute the cached readouts, at most every REFRESH_INTERVAL_S."""
    now = time.time()
    if not force and now - gui._statistics_time < REFRESH_INTERVAL_S:
        return
    gui._statistics_time = now
    gui.radio_map_stats = radio_map_statistics(gui)
    gui.link_budget_stats = link_budget(gui, *reversed(gui.cir_pair))


def noise_contents(gui: "SionnaRtGui") -> None:
    """
    The thermal noise floor, which is what turns received power into SNR and
    what the radio map's SINR is measured against. Sionna derives it from the
    scene's bandwidth and temperature as k * T * B.
    """
    bandwidth_mhz = float(gui.scene.bandwidth[0]) / 1e6
    property_row("Bandwidth [MHz]", gui.ui_scale)
    changed_bandwidth, new_bandwidth = psim.DragFloat(
        "##noise_bandwidth", bandwidth_mhz, 0.5, 0.1, 2000.0, format="%.1f"
    )
    end_property_row()

    temperature = float(gui.scene.temperature[0])
    property_row("Temperature [K]", gui.ui_scale)
    changed_temperature, new_temperature = psim.DragFloat(
        "##noise_temperature", temperature, 1.0, 1.0, 3000.0, format="%.0f"
    )
    end_property_row()

    if changed_bandwidth:
        gui.scene.bandwidth = new_bandwidth * 1e6
    if changed_temperature:
        gui.scene.temperature = new_temperature
    if changed_bandwidth or changed_temperature:
        # A radio map keeps the noise floor it was built with, so its SINR only
        # follows once it is recomputed.
        gui.reset_radio_map()
        refresh_statistics(gui, force=True)

    property_row("Receiver noise figure [dB]", gui.ui_scale)
    changed, gui.cfg.noise_figure_db = numeric_field(
        "##noise_figure", gui.cfg.noise_figure_db, 0.1, 0.0, 20.0, "%.1f"
    )
    end_property_row()
    if psim.IsItemHovered():
        psim.SetTooltip(
            "How much noise the receiver adds on top of the thermal floor.\n"
            "Around 7 dB is typical for a handset, 2-3 dB for a base station."
        )

    thermal_dbm = 10.0 * np.log10(float(gui.scene.thermal_noise_power[0])) + 30.0
    psim.Spacing()
    psim.TextColored(
        (*ACCENT_BRIGHT, 1.0),
        f"Noise floor {thermal_dbm + gui.cfg.noise_figure_db:.1f} dBm",
    )
    psim.PushTextWrapPos(0.0)
    psim.TextDisabled(
        f"Thermal k * T * B gives {thermal_dbm:.1f} dBm; the receiver figure adds "
        f"{gui.cfg.noise_figure_db:.1f} dB. Received power above this is the SNR. "
        "The radio map's SINR uses the thermal floor only, as sionna computes it."
    )
    psim.PopTextWrapPos()


def coverage_contents(gui: "SionnaRtGui") -> None:
    """Coverage statistics of the radio map, with a settable threshold."""
    if gui.radio_map is None:
        psim.TextDisabled("Compute a radio map to see coverage statistics.")
        return

    refresh_statistics(gui)
    stats = gui.radio_map_stats
    if stats is None:
        psim.TextDisabled("No statistics yet.")
        return
    if stats.get("covered", 0) == 0:
        psim.TextDisabled("The map has no coverage yet: let it refine.")
        return

    property_row("Coverage threshold [dBm]", gui.ui_scale)
    changed, gui.coverage_threshold_dbm = psim.SliderFloat(
        "##coverage_threshold", gui.coverage_threshold_dbm, -140.0, -20.0, format="%.0f"
    )
    end_property_row()

    rss = stats["rss_dbm_all"]
    above = float(np.count_nonzero(rss >= gui.coverage_threshold_dbm))
    served = 100.0 * above / max(stats["covered"], 1)
    of_area = 100.0 * above / max(stats["cells"], 1)

    psim.Spacing()
    psim.TextColored(
        (*ACCENT_BRIGHT, 1.0),
        f"{served:.1f}% of reached cells above threshold",
    )
    psim.TextDisabled(f"{of_area:.1f}% of the mapped area")
    psim.ProgressBar(served / 100.0, (-1.0, 0.0), f"{served:.1f}%")

    psim.Spacing()
    psim.PushTextWrapPos(0.0)
    psim.Text(
        f"Received power: median {stats['rss_dbm_median']:.1f} dBm, "
        f"mean {stats['rss_dbm_mean']:.1f} dBm"
    )
    psim.Text(
        f"Best cell {stats['rss_dbm_max']:.1f} dBm, "
        f"weakest tenth below {stats['rss_dbm_p10']:.1f} dBm"
    )
    psim.Text(
        f"SINR: median {stats['sinr_db_median']:.1f} dB, "
        f"mean {stats['sinr_db_mean']:.1f} dB"
    )
    psim.TextDisabled(
        f"{stats['covered']} of {stats['cells']} cells reached, "
        f"{stats['samples'] / 1e6:.1f}M samples accumulated"
    )
    psim.PopTextWrapPos()
    if psim.Button("Refresh##coverage"):
        refresh_statistics(gui, force=True)


def link_budget_contents(gui: "SionnaRtGui") -> None:
    """Link budget for the pair selected in the impulse response view."""
    if not gui.cfg.paths.compute_cir:
        psim.TextDisabled(
            "Enable the channel impulse response to get a link budget."
        )
        if psim.Button("Enable##link_budget"):
            gui.cfg.paths.compute_cir = True
            gui.update_paths(show=True)
        return

    refresh_statistics(gui)
    stats = gui.link_budget_stats
    if stats is None:
        psim.TextDisabled("No paths computed yet.")
        return
    if stats.get("paths", 0) == 0:
        psim.TextDisabled("No propagation paths between the selected pair.")
        return

    transmitters = list(gui.scene._transmitters.keys())
    receivers = list(gui.scene._receivers.keys())
    rx_index, tx_index = gui.cir_pair
    pair = (
        f"{transmitters[tx_index] if tx_index < len(transmitters) else tx_index}"
        f" -> "
        f"{receivers[rx_index] if rx_index < len(receivers) else rx_index}"
    )
    psim.PushTextWrapPos(0.0)
    psim.TextDisabled(pair)
    psim.TextColored(
        (*ACCENT_BRIGHT, 1.0),
        f"Received {stats['received_dbm']:.1f} dBm, SNR {stats['snr_db']:.1f} dB",
    )
    psim.TextDisabled(
        f"Noise floor {stats['noise_dbm']:.1f} dBm "
        f"({stats['thermal_dbm']:.1f} thermal + {stats['noise_figure_db']:.0f} dB "
        f"receiver figure)"
    )
    psim.Text(
        f"Path gain {stats['path_gain_db']:.1f} dB "
        f"from {stats['tx_power_dbm']:.1f} dBm transmitted"
    )
    psim.Text(
        f"{stats['paths']} path{'s' if stats['paths'] != 1 else ''}, "
        f"strongest {stats['strongest_db']:.1f} dB"
    )
    k_factor = stats["k_factor_db"]
    psim.Text(
        "Single path (no multipath)"
        if not np.isfinite(k_factor)
        else f"Dominant path {k_factor:.1f} dB above the rest"
    )
    psim.Text(
        f"Delay: mean {stats['mean_delay_ns']:.1f} ns, "
        f"spread {stats['delay_spread_ns']:.1f} ns"
    )
    if "doppler_max_hz" in stats:
        psim.Text(
            f"Doppler: up to {stats['doppler_max_hz']:.1f} Hz, "
            f"spread {stats['doppler_spread_hz']:.1f} Hz"
        )
    else:
        psim.TextDisabled("Doppler: devices are not moving")
    psim.PopTextWrapPos()
    if psim.Button("Refresh##link_budget"):
        refresh_statistics(gui, force=True)
