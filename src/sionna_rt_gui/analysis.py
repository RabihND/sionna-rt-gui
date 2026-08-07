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

from .ps_utils import ACCENT_BRIGHT, im_col32
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


def link_simulation(gui: "SionnaRtGui", rx_index: int = 0, tx_index: int = 0) -> dict | None:
    """
    Quality of the traced channel across an OFDM grid: the frequency response
    from the path coefficients, the signal-to-noise ratio per subcarrier against
    the noise floor, and the resulting Shannon capacity.

    These are definitions rather than a simulated transmission. Sending symbols,
    coding them and counting errors is the job of sionna's physical layer
    package, which is not installed alongside sionna-rt.
    """
    if gui.paths_cir is None:
        return None
    a, tau = gui.paths_cir
    if a.shape[0] <= rx_index or a.shape[2] <= tx_index:
        return None

    coefficients = a[rx_index, 0, tx_index, 0, :, 0]
    delays = (
        tau[rx_index, tx_index, :] if tau.ndim == 3 else tau[rx_index, 0, tx_index, 0, :]
    )
    valid = (delays >= 0.0) & (np.abs(coefficients) > 0.0)
    coefficients, delays = coefficients[valid], delays[valid]
    if coefficients.size == 0:
        return {"paths": 0}

    paths_cfg = gui.cfg.paths
    n_subcarriers = min(max(int(paths_cfg.fft_size), 64), 2048)
    spacing = float(paths_cfg.subcarrier_spacing)
    bandwidth = n_subcarriers * spacing

    # Frequency response of the traced channel across the OFDM grid
    frequencies = (np.arange(n_subcarriers) - n_subcarriers // 2) * spacing
    response = np.exp(
        -2j * np.pi * frequencies[:, None] * delays[None, :]
    ) @ coefficients

    # Powers per subcarrier: the transmitter's power and the noise floor are
    # both spread over the grid
    tx_power_w = 10.0 ** (float(gui.link_budget_tx_power_dbm(tx_index)) / 10.0) / 1000.0
    thermal_w = float(gui.scene.thermal_noise_power[0]) * 10.0 ** (
        gui.cfg.noise_figure_db / 10.0
    )
    signal = tx_power_w / n_subcarriers * np.abs(response) ** 2
    noise_per_subcarrier = thermal_w / n_subcarriers
    snr = signal / max(noise_per_subcarrier, 1e-30)

    capacity_bps = float(np.sum(np.log2(1.0 + snr)) * spacing)
    return {
        "snr_linear": snr,
        "response_db": 20.0 * np.log10(np.maximum(np.abs(response), 1e-12)),
        "paths": int(coefficients.size),
        "subcarriers": n_subcarriers,
        "bandwidth_mhz": bandwidth / 1e6,
        "snr_mean_db": float(10.0 * np.log10(np.mean(snr))),
        "snr_min_db": float(10.0 * np.log10(np.min(snr))),
        "snr_max_db": float(10.0 * np.log10(np.max(snr))),
        "flatness_db": float(
            10.0 * np.log10(np.max(np.abs(response) ** 2) / np.min(np.abs(response) ** 2))
        ),
        "capacity_mbps": capacity_bps / 1e6,
    }


def refresh_statistics(gui: "SionnaRtGui", force: bool = False) -> None:
    """Recompute the cached readouts, at most every REFRESH_INTERVAL_S."""
    now = time.time()
    if not force and now - gui._statistics_time < REFRESH_INTERVAL_S:
        return
    gui._statistics_time = now
    gui.radio_map_stats = radio_map_statistics(gui)
    gui.link_budget_stats = link_budget(gui, *reversed(gui.cir_pair))
    gui.link_simulation_stats = link_simulation(gui, *reversed(gui.cir_pair))


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


def phy_available() -> bool:
    """Whether sionna's physical layer package is installed alongside sionna-rt."""
    try:
        import sionna.phy  # noqa: F401
    except Exception:
        return False
    return True


def link_simulation_contents(gui: "SionnaRtGui") -> None:
    """Channel quality across the band, and what a full link would need."""
    if not gui.cfg.paths.compute_cir:
        psim.TextDisabled("Enable the channel impulse response first.")
        return
    refresh_statistics(gui)
    stats = gui.link_simulation_stats
    if stats is None or stats.get("paths", 0) == 0:
        psim.TextDisabled("No paths to transmit through.")
        return

    psim.PushTextWrapPos(0.0)
    psim.TextColored(
        (*ACCENT_BRIGHT, 1.0),
        f"{stats['capacity_mbps']:.1f} Mbit/s over {stats['bandwidth_mhz']:.1f} MHz",
    )
    psim.TextDisabled("Shannon capacity of the traced channel with this noise floor")
    psim.Spacing()
    psim.Text(
        f"SNR per subcarrier: mean {stats['snr_mean_db']:.1f} dB, "
        f"{stats['snr_min_db']:.1f} to {stats['snr_max_db']:.1f} dB"
    )
    psim.Text(f"Frequency selectivity: {stats['flatness_db']:.1f} dB across the band")
    psim.Spacing()
    if phy_available():
        psim.TextDisabled(
            "Sionna's physical layer package is available, so a standards-based "
            "link (5G NR or generic OFDM) can be run from these channels."
        )
    else:
        psim.TextDisabled(
            "Bit and block error rates need a waveform, coding and a receiver, "
            "which live in sionna's physical layer package. Only sionna-rt is "
            "installed here, so the figures above stop at the channel itself."
        )
    psim.PopTextWrapPos()


def _flatness_note(channel: dict) -> str:
    """How even the channel is across the band, in one clause."""
    if channel["flatness_db"] > 1.0:
        return (
            f"the channel varies by {channel['flatness_db']:.1f} dB across the "
            "band"
        )
    return f"the channel is flat across the band ({channel['flatness_db']:.1f} dB)"


def phy_link_contents(gui: "SionnaRtGui") -> None:
    """
    Link metrics from sionna's system-level layer, with the per-subcarrier
    signal-to-noise ratio drawn across the band.
    """
    from . import phy_metrics

    scale = gui.ui_scale
    if not gui.cfg.paths.compute_cir:
        psim.TextDisabled("Enable the channel impulse response to measure a link.")
        if psim.Button("Enable##phy_link"):
            gui.cfg.paths.compute_cir = True
            gui.update_paths(show=True)
        return

    refresh_statistics(gui)
    channel = gui.link_simulation_stats
    if channel is None or channel.get("paths", 0) == 0:
        psim.TextDisabled("No paths between the selected pair yet.")
        return

    if not phy_metrics.available():
        psim.PushTextWrapPos(0.0)
        psim.TextDisabled(
            "Block error rates, throughput and modulation choice come from "
            "sionna's system-level layer, which arrives with the full sionna "
            "package. Only sionna-rt is installed, so the channel figures in "
            "the Analysis tab are all that can be shown."
        )
        psim.PopTextWrapPos()
        return

    # --- Controls and the measurement itself
    psim.AlignTextToFramePadding()
    psim.Text("Target block error rate")
    psim.SameLine()
    psim.PushItemWidth(120 * scale)
    _, gui.phy_bler_target = psim.SliderFloat(
        "##phy_bler_target", gui.phy_bler_target, 0.01, 0.5, format="%.2f"
    )
    psim.PopItemWidth()
    psim.SameLine()
    if psim.Button("Measure##phy_link") or (
        gui.phy_metrics_stats is None and gui.phy_auto_measure
    ):
        gui.phy_auto_measure = False
        # The channel figures are refreshed on a timer; a measurement has to be
        # made against the channel as it stands now, not as it stood a moment ago
        refresh_statistics(gui, force=True)
        channel = gui.link_simulation_stats or channel
        gui.phy_metrics_stats = phy_metrics.link_metrics(
            channel["snr_linear"], bler_target=gui.phy_bler_target
        )
        gui.link_metrics_time = time.time()
    psim.SameLine()
    psim.TextDisabled("over sionna's 3GPP tables")
    psim.SameLine()
    if psim.Button("Run 5G NR##phy_nr"):
        gui.nr_link_stats = _run_nr_link(gui)
        gui.link_metrics_time = time.time()
    if psim.IsItemHovered():
        psim.SetTooltip(
            "Send coded 5G NR uplink slots through this channel and count the\n"
            "bit errors. Uses sionna's waveform, pilots, coding and receiver."
        )

    stats = gui.phy_metrics_stats
    available_width = psim.GetContentRegionAvail()[0]

    # Results describe the geometry they were measured on
    is_stale = (
        stats is not None
        and gui._last_paths_update_time > gui.link_metrics_time + 1e-6
    )
    if is_stale:
        psim.SameLine()
        psim.TextColored(
            (0.92, 0.55, 0.30, 1.0), "- the devices have moved since"
        )

    if stats is not None:
        chosen = stats["chosen"]
        # Roughly one slot of the configured numerology
        slot_seconds = stats["num_ofdm_symbols"] / max(
            float(gui.cfg.paths.subcarrier_spacing), 1.0
        )
        offered_mbps = chosen["bits_per_block"] / slot_seconds / 1e6
        delivered_mbps = offered_mbps * (1.0 - min(max(chosen["bler"], 0.0), 1.0))
        psim.Spacing()
        psim.TextColored(
            (*ACCENT_BRIGHT, 1.0),
            f"MCS {chosen['mcs_index']} delivers {delivered_mbps:.1f} Mbit/s at "
            f"{chosen['bler']:.1%} block error rate",
        )
        psim.Text(
            f"Effective SINR {chosen['sinr_eff_db']:.1f} dB  ·  "
            f"{stats['num_subcarriers']} subcarriers  ·  "
            f"{chosen['bits_per_block']} bits per block  ·  "
            f"Shannon capacity {channel['capacity_mbps']:.1f} Mbit/s"
        )
        if not stats["meets_target"]:
            psim.TextColored(
                (0.92, 0.55, 0.30, 1.0),
                "No scheme meets the target: the link would not close",
            )
        # Everything worth knowing but not worth a line of its own
        notes = [_flatness_note(channel)]
        if chosen["sinr_eff_db"] >= 29.9:
            notes.append("the tables stop at 30 dB, so a stronger link reads the same")
        missing = stats.get("unavailable") or []
        if len(missing) > 1:
            notes.append(
                f"MCS {missing[0]}-{missing[-1]} are not in sionna's table here"
            )
        elif missing:
            notes.append(f"MCS {missing[0]} is not in sionna's table here")
        psim.TextDisabled("  ·  ".join(notes))
    else:
        psim.TextDisabled(_flatness_note(channel))

    nr = gui.nr_link_stats
    if nr is not None:
        psim.Spacing()
        if "error" in nr:
            psim.TextColored((0.92, 0.45, 0.40, 1.0), f"5G NR link failed: {nr['error']}")
        else:
            delivered = nr["blocks_delivered"]
            if delivered:
                psim.TextColored(
                    (*ACCENT_BRIGHT, 1.0),
                    f"5G NR uplink: {nr['throughput_mbps']:.2f} Mbit/s, "
                    f"bit error rate {nr['ber']:.2e}",
                )
            else:
                psim.TextColored(
                    (0.92, 0.55, 0.30, 1.0),
                    "5G NR uplink: nothing delivered, every block failed its check",
                )
            psim.TextDisabled(
                f"{delivered} of {nr['blocks']} transport block(s) passed, "
                f"{nr['bit_errors']} of {nr['bits']} coded bits wrong, "
                f"{nr['subcarriers']} subcarriers at "
                f"{nr['subcarrier_spacing_khz']:.0f} kHz, per-element SNR "
                f"{nr['snr_db']:.1f} dB"
            )
    else:
        psim.Spacing()
        psim.TextDisabled(
            "Run 5G NR to send coded slots through this channel and count the "
            "bit errors."
        )

    # --- Why this scheme was chosen: what every scheme would deliver here
    if stats is None:
        psim.Spacing()
        psim.TextDisabled(
            "Measure the link to see how much each modulation and coding "
            "scheme would deliver on this channel."
        )
        return

    schemes = sorted(stats["all"], key=lambda r: r["mcs_index"])
    slot_seconds = stats["num_ofdm_symbols"] / max(
        float(gui.cfg.paths.subcarrier_spacing), 1.0
    )
    # A block that fails its check carries nothing, so the bars show what gets
    # through and the errors take their share off the top of each bar.
    offered = np.array([r["bits_per_block"] / slot_seconds / 1e6 for r in schemes])
    blers = np.clip([r["bler"] for r in schemes], 0.0, 1.0)
    delivered = offered * (1.0 - blers)
    peak = float(np.max(offered)) if offered.size else 0.0

    psim.Spacing()
    if peak <= 0.0:
        psim.TextDisabled(
            "Nothing gets through: every scheme in the table loses all its "
            "blocks at this signal-to-noise ratio, so there is nothing to plot."
        )
        return

    psim.TextDisabled(
        "One bar per modulation and coding scheme, slower and more robust on "
        "the left. Bar height is what the scheme delivers here."
    )

    draw_list = psim.GetWindowDrawList()
    origin = psim.GetCursorScreenPos()
    available_height = psim.GetContentRegionAvail()[1]
    plot_height = max(available_height - 40 * scale, 96 * scale)
    plot_width = max(available_width - 60 * scale, 120 * scale)
    x0 = origin[0] + 52 * scale
    y0 = origin[1] + 4 * scale

    grid_color = im_col32(0.55, 0.57, 0.58, 0.22)
    label_color = im_col32(0.62, 0.64, 0.65, 0.9)
    bar_color = im_col32(0.28, 0.45, 0.70)
    lost_color = im_col32(0.62, 0.32, 0.30)
    axis_format = "%.0f" if peak >= 10.0 else ("%.1f" if peak >= 1.0 else "%.2f")

    # Gridlines, in the single unit the bars are drawn in
    for fraction in (0.0, 0.5, 1.0):
        y = y0 + fraction * plot_height
        draw_list.AddLine((x0, y), (x0 + plot_width, y), grid_color, 1.0)
        value = axis_format % (peak * (1.0 - fraction))
        # The top of the scale carries the unit, just under its line so it has
        # the room; the shortest bars are on that side
        draw_list.AddText(
            (origin[0], y + (2 if fraction == 0.0 else -7) * scale),
            label_color,
            f"{value} Mbit/s" if fraction == 0.0 else value,
        )

    step = plot_width / max(len(schemes), 1)
    chosen_index = stats["chosen"]["mcs_index"]
    mouse_x, mouse_y = psim.GetMousePos()
    hovered = None
    for i, scheme in enumerate(schemes):
        left = x0 + i * step + 0.15 * step
        right = x0 + (i + 1) * step - 0.15 * step
        top_offered = y0 + (1.0 - offered[i] / peak) * plot_height
        top_delivered = y0 + (1.0 - delivered[i] / peak) * plot_height
        is_chosen = scheme["mcs_index"] == chosen_index
        # What the errors take, above what survives them
        if top_delivered - top_offered > 1.0:
            draw_list.AddRectFilled(
                (left, top_offered), (right, top_delivered), lost_color
            )
        draw_list.AddRectFilled(
            (left, top_delivered),
            (right, y0 + plot_height),
            im_col32(*ACCENT_BRIGHT) if is_chosen else bar_color,
        )
        if i % 4 == 0 or is_chosen:
            draw_list.AddText(
                (left, y0 + plot_height + 3 * scale),
                im_col32(*ACCENT_BRIGHT) if is_chosen else label_color,
                str(scheme["mcs_index"]),
            )
        if left - 0.15 * step <= mouse_x <= right + 0.15 * step:
            hovered = (scheme, offered[i], delivered[i])

    # Each bar explains itself on hover
    if (
        hovered is not None
        and psim.IsWindowHovered()
        and y0 <= mouse_y <= y0 + plot_height
    ):
        scheme, scheme_offered, scheme_delivered = hovered
        psim.SetTooltip(
            f"MCS {scheme['mcs_index']}\n"
            f"{scheme_delivered:.1f} Mbit/s delivered of {scheme_offered:.1f} "
            f"Mbit/s attempted\n"
            f"{scheme['bler']:.1%} of blocks fail their check\n"
            f"{scheme['bits_per_block']} bits per block"
        )

    legend_y = y0 + plot_height + 15 * scale
    draw_list.AddRectFilled(
        (x0, legend_y + 3 * scale),
        (x0 + 9 * scale, legend_y + 12 * scale),
        bar_color,
    )
    draw_list.AddText(
        (x0 + 14 * scale, legend_y), label_color, "delivered"
    )
    lost_x = x0 + 90 * scale
    draw_list.AddRectFilled(
        (lost_x, legend_y + 3 * scale),
        (lost_x + 9 * scale, legend_y + 12 * scale),
        lost_color,
    )
    draw_list.AddText(
        (lost_x + 14 * scale, legend_y),
        label_color,
        f"lost to block errors        the fastest scheme staying under the "
        f"{gui.phy_bler_target:.0%} target is used",
    )
    psim.Dummy((available_width, plot_height + 40 * scale))


def _run_nr_link(gui: "SionnaRtGui") -> dict | None:
    """Feed the traced channel of the selected pair into sionna's 5G NR link."""
    from . import phy_metrics

    if gui.paths is None:
        return {"error": "no paths computed"}
    a, tau = gui.paths.cir(out_type="numpy", normalize_delays=True)
    rx_index, tx_index = gui.cir_pair
    # One pair at a time, with the axes the physical layer expects
    a_pair = a[rx_index : rx_index + 1, :, tx_index : tx_index + 1, ...]
    tau_pair = (
        tau[rx_index : rx_index + 1, tx_index : tx_index + 1, ...]
        if tau.ndim == 3
        else tau[rx_index : rx_index + 1, :, tx_index : tx_index + 1, ...]
    )
    tx_power_w = 10.0 ** (gui.link_budget_tx_power_dbm(tx_index) / 10.0) / 1000.0
    noise_w = float(gui.scene.thermal_noise_power[0]) * 10.0 ** (
        gui.cfg.noise_figure_db / 10.0
    )
    return phy_metrics.nr_link(a_pair, tau_pair, tx_power_w, noise_w)
