#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Link metrics from sionna's system-level layer, driven by the traced channel.

sionna-rt gives the channel; turning per-subcarrier signal-to-noise ratios into
a block error rate, a throughput and a suitable modulation and coding scheme is
what ``sionna.sys.PHYAbstraction`` does, against the 3GPP tables it ships. That
is used here rather than any home-made receiver, so the figures are sionna's.

The package is optional: it arrives with ``sionna`` (which also brings
TensorFlow) while the viewer only requires ``sionna-rt``. It is therefore
imported lazily, and TensorFlow is kept off the GPU so it cannot compete with
the ray tracer for memory.
"""
from __future__ import annotations

import logging

import numpy as np

# Filled in on first use
_phy_abstraction = None
_import_error: str | None = None
_available: bool | None = None

# Modulation and coding schemes of the 3GPP table used below
MCS_TABLE_INDEX = 1
MCS_INDICES = tuple(range(28))


def available() -> bool:
    """Whether sionna's system-level layer can be used."""
    global _available, _import_error
    if _available is not None:
        return _available
    try:
        import sionna.sys  # noqa: F401

        _available = True
    except Exception as e:  # pragma: no cover - depends on the environment
        _import_error = str(e)
        _available = False
    return _available


def import_error() -> str | None:
    return _import_error


def _prepare_tensorflow() -> None:
    """
    Keep TensorFlow on the CPU. The ray tracer owns the GPU, and TensorFlow
    reserves most of its memory on first use, which would starve it. These link
    calculations are small, so the CPU is ample.
    """
    import tensorflow as tf

    try:
        if tf.config.list_physical_devices("GPU"):
            tf.config.set_visible_devices([], "GPU")
    except Exception as e:  # pragma: no cover
        logging.debug("Could not confine TensorFlow to the CPU: %s", e)


def _abstraction():
    global _phy_abstraction
    if _phy_abstraction is None:
        _prepare_tensorflow()
        from sionna.sys import PHYAbstraction

        _phy_abstraction = PHYAbstraction()
    return _phy_abstraction


def link_metrics(
    sinr_per_subcarrier: np.ndarray,
    num_ofdm_symbols: int = 14,
    mcs_index: int | None = None,
    bler_target: float = 0.1,
) -> dict | None:
    """
    Effective SINR, block error rate and throughput for a channel described by
    its per-subcarrier signal-to-noise ratios (linear, not dB).

    With no `mcs_index`, every scheme in the table is evaluated and the most
    efficient one whose block error rate stays under `bler_target` is chosen,
    which is what link adaptation does.
    """
    if not available():
        return None
    sinr_per_subcarrier = np.asarray(sinr_per_subcarrier, dtype=np.float64)
    if sinr_per_subcarrier.size == 0:
        return None

    import tensorflow as tf

    abstraction = _abstraction()

    # PHYAbstraction expects [..., symbols, subcarriers, users, streams]
    grid = np.tile(
        sinr_per_subcarrier[None, :, None, None],
        (num_ofdm_symbols, 1, 1, 1),
    )
    sinr = tf.constant(grid, dtype=tf.float32)

    candidates = (mcs_index,) if mcs_index is not None else MCS_INDICES
    results = []
    unavailable = []
    for index in candidates:
        try:
            decoded_bits, _, sinr_eff, tbler, bler = abstraction(
                mcs_index=tf.constant([index], dtype=tf.int32),
                sinr=sinr,
                mcs_table_index=MCS_TABLE_INDEX,
            )
        except Exception as e:
            logging.debug("MCS %s could not be evaluated: %s", index, e)
            unavailable.append(int(index))
            continue
        block_error_rate = float(np.ravel(tbler.numpy())[0])
        # The shipped tables do not cover every scheme of every table index;
        # those come back without a finite error rate and are not results.
        if not np.isfinite(block_error_rate):
            unavailable.append(int(index))
            continue
        results.append(
            {
                "mcs_index": int(index),
                "sinr_eff_db": float(
                    10.0 * np.log10(max(float(np.ravel(sinr_eff.numpy())[0]), 1e-12))
                ),
                "bler": block_error_rate,
                "bits_per_block": int(np.ravel(decoded_bits.numpy())[0]),
            }
        )

    if not results:
        return None

    # Highest scheme meeting the target, else the most robust one
    within_target = [r for r in results if 0.0 <= r["bler"] <= bler_target]
    chosen = (
        max(within_target, key=lambda r: r["mcs_index"])
        if within_target
        else min(results, key=lambda r: r["bler"])
    )
    return {
        "chosen": chosen,
        "all": results,
        "unavailable": sorted(set(unavailable)),
        "meets_target": bool(within_target),
        "bler_target": bler_target,
        "num_ofdm_symbols": num_ofdm_symbols,
        "num_subcarriers": int(sinr_per_subcarrier.size),
    }


# Built once: constructing a transmitter and receiver is slow, running them is not
_nr_link = None


def _nr_components():
    """The 5G NR uplink transmitter and receiver, built once."""
    global _nr_link
    if _nr_link is None:
        _prepare_tensorflow()
        from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, PUSCHTransmitter

        config = PUSCHConfig()
        transmitter = PUSCHTransmitter(config, verbose=False)
        # The transport block's check tells us whether the slot was really
        # delivered; a bit error rate alone would suggest throughput on a link
        # that in fact carries nothing.
        receiver = PUSCHReceiver(transmitter, return_tb_crc_status=True)
        _nr_link = (transmitter, receiver)
    return _nr_link


def nr_link(
    a: np.ndarray,
    tau: np.ndarray,
    tx_power_w: float,
    noise_power_w: float,
    num_slots: int = 1,
) -> dict | None:
    """
    Send 5G NR uplink slots through the traced channel and count the bit errors.

    The channel comes from the ray tracer, the waveform, coding, pilots and
    receiver from sionna's physical layer. Coefficients travel through host
    memory rather than the device, so TensorFlow never has to share the GPU with
    the tracer.
    """
    if not available():
        return None
    try:
        import tensorflow as tf
        from sionna.phy.channel import (
            ApplyOFDMChannel,
            cir_to_ofdm_channel,
            subcarrier_frequencies,
        )

        transmitter, receiver = _nr_components()
        grid = transmitter.resource_grid
        frequencies = subcarrier_frequencies(grid.fft_size, grid.subcarrier_spacing)

        # [batch, rx, rx_ant, tx, tx_ant, paths, time] as the physical layer expects
        a_tf = tf.constant(np.asarray(a)[None, ...], dtype=tf.complex64)
        tau_tf = tf.constant(np.asarray(tau)[None, ...], dtype=tf.float32)
        h_freq = cir_to_ofdm_channel(frequencies, a_tf, tau_tf, normalize=False)

        # Absolute powers, spread over the grid
        per_subcarrier_tx = max(tx_power_w, 1e-30) / grid.fft_size
        per_subcarrier_no = max(noise_power_w, 1e-30) / grid.fft_size
        h_scaled = tf.cast(np.sqrt(per_subcarrier_tx), tf.complex64) * h_freq
        no = tf.constant(per_subcarrier_no, tf.float32)

        apply_channel = ApplyOFDMChannel()
        errors = 0
        total = 0
        blocks = 0
        blocks_delivered = 0
        for _ in range(max(int(num_slots), 1)):
            x, bits = transmitter(1)
            y = apply_channel(x, h_scaled, no)
            bits_hat, crc_ok = receiver(y, no)
            errors += int(
                tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), tf.int32)).numpy()
            )
            total += int(np.prod(bits.shape))
            status = np.asarray(crc_ok.numpy()).ravel()
            blocks += status.size
            blocks_delivered += int(np.count_nonzero(status))

        mean_gain = float(tf.reduce_mean(tf.abs(h_freq) ** 2))
        snr_db = 10.0 * np.log10(
            max(per_subcarrier_tx * mean_gain / per_subcarrier_no, 1e-30)
        )
        slot_seconds = grid.num_ofdm_symbols / max(grid.subcarrier_spacing, 1.0)
        bits_per_slot = total / max(int(num_slots), 1)
        # Only blocks that pass their check carry data
        delivered_fraction = blocks_delivered / max(blocks, 1)
        return {
            "ber": errors / max(total, 1),
            "bit_errors": errors,
            "bits": total,
            "slots": int(num_slots),
            "blocks": blocks,
            "blocks_delivered": blocks_delivered,
            "bler": 1.0 - delivered_fraction,
            "snr_db": snr_db,
            "bits_per_slot": int(bits_per_slot),
            "throughput_mbps": bits_per_slot * delivered_fraction / slot_seconds / 1e6,
            "subcarriers": int(grid.fft_size),
            "ofdm_symbols": int(grid.num_ofdm_symbols),
            "subcarrier_spacing_khz": grid.subcarrier_spacing / 1e3,
        }
    except Exception as e:
        logging.error("The 5G NR link could not be run: %s", e)
        return {"error": str(e)}
