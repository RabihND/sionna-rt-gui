from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from typing import Iterator

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from .animation import animation_tick_single_object
from .sionna_utils import prepare_and_normalize_cir


class CirBatchExporter(Iterator[int]):
    def __init__(
        self,
        gui: "SionnaRtGui",
        tx_index: int,
        rx_index: int,
        output_filename: str,
        duration_s: float,
        sampling_frequency_hz: float,
        interpolation_factor: int,
    ):
        self.main = gui
        self.tx_index = tx_index
        self.tx_name = list(gui.scene._transmitters.keys())[tx_index]
        self.rx_index = rx_index
        self.rx_name = list(gui.scene._receivers.keys())[rx_index]
        self.output_filename = output_filename
        self.duration_s = duration_s
        self.sampling_frequency_hz = sampling_frequency_hz
        self.interpolation_factor = interpolation_factor

        # TODO: create temporary UEs for parallelisation

        # Save all simulation parameters to a JSON file.
        n_cirs = int(np.ceil(duration_s * sampling_frequency_hz))
        with open(output_filename, "w") as f:
            json.dump(
                {
                    "batch": {
                        "tx_index": tx_index,
                        "tx_name": self.tx_name,
                        "rx_index": rx_index,
                        "rx_name": self.rx_name,
                        "sampling_frequency_hz": sampling_frequency_hz,
                        "interpolation_factor": interpolation_factor,
                        "n_cirs": n_cirs,
                        "duration_s": duration_s,
                    },
                    "paths": asdict(gui.cfg.paths),
                    "srk": asdict(gui.cfg.srk_demo),
                },
                f,
                indent=4,
            )

        # Iterator state
        self.cir_i = 0
        self.n_cirs = n_cirs
        self.is_cancelled = False

        # We compute one true CIR every `interpolation_factor` time steps, and interpolate
        # the rest using Doppler.
        # TODO: double-check that the Doppler vector is set correctly.
        self.time_delta = interpolation_factor / sampling_frequency_hz
        self.num_taps = gui.cfg.paths.num_taps
        self.output_bin_fname = os.path.splitext(output_filename)[0] + ".bin"

    def __iter__(self):
        return self

    def __next__(self):
        """
        Export one CIR for each OFDM symbol in the given duration.
        Advance the animation as we go.
        """

        if self.cir_i >= self.n_cirs or self.is_cancelled:
            # Note: we leave all animations paused even if they were playing before.
            raise StopIteration

        mode = "ab" if self.cir_i > 0 else "wb"
        with open(self.output_bin_fname, mode) as f:
            with replace_radio_devices(self.main.scene, self.tx_index, self.rx_index):
                # Advance animation
                # TODO: advance animation for all instances of the RX (parallelization)
                for name in (self.tx_name, self.rx_name):
                    animation_tick_single_object(
                        self.main, name, time_delta=self.time_delta, force=True
                    )

                # Compute updated paths
                paths = self.main.compute_paths()
                if paths is None:
                    # TODO: still need to write a CIR
                    raise NotImplementedError("No paths available")

                paths_cfg = self.main.cfg.paths
                paths_taps = paths.taps(
                    bandwidth=paths_cfg.bandwidth,
                    l_min=paths_cfg.l_min,
                    l_max=paths_cfg.l_max,
                    sampling_frequency=paths_cfg.sampling_frequency,
                    num_time_steps=self.interpolation_factor,
                    normalize=paths_cfg.normalize,
                    normalize_delays=paths_cfg.normalize_delays,
                    out_type="numpy",
                )

                taps_results = prepare_and_normalize_cir(
                    taps=paths_taps,
                    # TODO: adjust this if we manipulate which radio devices are part of the scene
                    tx_index=self.tx_index,
                    rx_index=self.rx_index,
                    num_taps=self.num_taps,
                    bandwidth=paths_cfg.bandwidth,
                    snr_offset_db=paths_cfg.snr_offset_db,
                    max_noise_std=self.main.cfg.srk_demo.max_noise_std,
                    as_arrays=True,
                )

            taps_norm = taps_results["taps_norm"]
            taps = taps_results["taps"]
            assert taps_norm.shape == (self.interpolation_factor,)
            assert taps.shape == (self.interpolation_factor, self.num_taps * 2)
            assert taps_norm.dtype == np.float32
            assert taps.dtype == np.float32

            # Strictly respect the CIR count even if not a multiple of the interpolation factor.
            if self.cir_i + self.interpolation_factor >= self.n_cirs:
                taps_norm = taps_norm[: self.n_cirs - self.cir_i]
                taps = taps[: self.n_cirs - self.cir_i, :]

            # For each CIR (including the interpolated ones), write:
            #     norm, tap0_real, tap0_imag, tap1_real, tap1_imag, ..., tapN_real, tapN_imag
            for k in range(taps.shape[0]):
                f.write(taps_norm[k].tobytes())
                f.write(taps[k, :].tobytes())

            self.cir_i += self.interpolation_factor

    def gui(self):
        ui_scale = self.main.ui_scale
        window_resolution = ps.get_window_size()
        w, h = 600, 100
        psim.SetNextWindowSize((w * ui_scale, h * ui_scale))
        psim.SetNextWindowPos(
            (
                0.5 * (window_resolution[0] - w * ui_scale),
                0.5 * (window_resolution[1] - h * ui_scale),
            )
        )

        _ = psim.Begin(
            "CIR batch export",
            open=True,
            flags=psim.ImGuiWindowFlags_Modal | psim.ImGuiWindowFlags_NoDecoration,
        )

        label = f"Exporting CIRs ({self.cir_i}/{self.n_cirs})..."
        psim.Text(label)

        psim.ProgressBar(min(self.cir_i / self.n_cirs, 1.0))

        bw = 100 * ui_scale
        psim.SetCursorPosX((w - 10) * ui_scale - bw)
        if psim.Button("Cancel##cir_batch_export", size=(bw, 0)):
            self.is_cancelled = True

        psim.End()


@contextmanager
def replace_radio_devices(scene: rt.Scene, tx_index: int, rx_index: int):
    tx_bak = scene.transmitters.copy()
    rx_bak = scene.receivers.copy()
    scene._transmitters = {
        tx_bak[k].name: tx_bak[k] for i, k in enumerate(tx_bak.keys()) if i == tx_index
    }
    scene._receivers = {
        rx_bak[k].name: rx_bak[k] for i, k in enumerate(rx_bak.keys()) if i == rx_index
    }

    yield

    scene._transmitters = tx_bak
    scene._receivers = rx_bak


def export_cir_batch(
    gui: "SionnaRtGui",
    tx_index: int,
    rx_index: int,
    output_filename: str,
    duration_s: float,
    sampling_frequency_hz: float,
    interpolation_factor: int,
) -> CirBatchExporter | None:
    if tx_index >= len(gui.scene._transmitters) or rx_index >= len(
        gui.scene._receivers
    ):
        return None

    # We will advance animations manually.
    gui.set_all_animations_playing(False)
    gui.cfg.paths.auto_update = False
    gui.cfg.radio_map.auto_update = False

    # TODO: auto-cancel if there's no animation.
    return CirBatchExporter(
        gui,
        tx_index,
        rx_index,
        output_filename,
        duration_s,
        sampling_frequency_hz,
        interpolation_factor,
    )
