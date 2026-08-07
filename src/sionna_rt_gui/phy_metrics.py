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
    for index in candidates:
        try:
            decoded_bits, _, sinr_eff, tbler, bler = abstraction(
                mcs_index=tf.constant([index], dtype=tf.int32),
                sinr=sinr,
                mcs_table_index=MCS_TABLE_INDEX,
            )
        except Exception as e:
            logging.debug("MCS %s could not be evaluated: %s", index, e)
            continue
        results.append(
            {
                "mcs_index": int(index),
                "sinr_eff_db": float(
                    10.0 * np.log10(max(float(np.ravel(sinr_eff.numpy())[0]), 1e-12))
                ),
                "bler": float(np.ravel(tbler.numpy())[0]),
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
        "meets_target": bool(within_target),
        "bler_target": bler_target,
        "num_ofdm_symbols": num_ofdm_symbols,
        "num_subcarriers": int(sinr_per_subcarrier.size),
    }
