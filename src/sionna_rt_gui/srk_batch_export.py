from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from copy import copy, deepcopy
import json
import os
from typing import Iterator

import drjit as dr
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from .sionna_utils import prepare_and_normalize_cir_batch


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
        parallelism: int,
    ):
        self.main = gui
        self.output_filename: str = output_filename
        self.duration_s: float = duration_s
        self.sampling_frequency_hz: float = sampling_frequency_hz
        self.interpolation_factor: int = interpolation_factor
        self.parallelism: int = parallelism
        assert parallelism >= 1

        # Iterator state
        self.cir_i = 0
        self.n_cirs = int(np.ceil(duration_s * sampling_frequency_hz))
        self.n_cirs_written = 0
        self.is_cancelled = False

        # We compute one true CIR every `interpolation_factor` time steps, and interpolate
        # the rest using Doppler.
        self.time_delta = interpolation_factor / sampling_frequency_hz
        self.num_taps = gui.cfg.paths.num_taps
        self.output_bin_fname = os.path.splitext(output_filename)[0] + ".bin"

        self.devices = self._prepare_temporary_radio_devices(tx_index, rx_index)

        # Save all simulation parameters to a JSON file.
        with open(output_filename, "w") as f:
            json.dump(
                {
                    "batch": {
                        "tx_index": tx_index,
                        "rx_index": rx_index,
                        "sampling_frequency_hz": sampling_frequency_hz,
                        "interpolation_factor": interpolation_factor,
                        "n_cirs": self.n_cirs,
                        "duration_s": duration_s,
                    },
                    "paths": asdict(gui.cfg.paths),
                    "srk": asdict(gui.cfg.srk_demo),
                },
                f,
                indent=4,
            )

    def __iter__(self):
        return self

    def __next__(self):
        """
        Export one CIR for each OFDM symbol in the given duration.
        Advance the animation as we go.

        We accelerate this computation in two ways:
        - We instantiate `self.parallelism` copies of the radio devices of interest
          and place them along the trajectories, so that their CIRs can be computed in parallel.
        - At each position along the trajectory, we extrapolate CIRs using Doppler.
        """

        if self.cir_i >= self.n_cirs or self.is_cancelled:
            # Note: we leave all animations paused even if they were playing before.
            if not self.is_cancelled:
                assert self.n_cirs_written == self.n_cirs
            raise StopIteration

        paths_cfg = self.main.cfg.paths
        tx_instances = self.devices["tx"]["instances"]
        rx_instances = self.devices["rx"]["instances"]

        with replace_radio_devices(
            self.main.scene,
            tx_instances,
            rx_instances,
        ):
            # Compute updated paths
            solver = rt.PathSolver()
            paths = solver(
                self.main.scene,
                max_depth=paths_cfg.max_depth,
                max_num_paths_per_src=paths_cfg.max_num_paths_per_src,
                samples_per_src=paths_cfg.samples_per_src,
                synthetic_array=paths_cfg.synthetic_array,
                los=paths_cfg.los,
                specular_reflection=paths_cfg.specular_reflection,
                diffuse_reflection=paths_cfg.diffuse_reflection,
                refraction=paths_cfg.refraction,
                diffraction=paths_cfg.diffraction,
                edge_diffraction=paths_cfg.edge_diffraction,
                diffraction_lit_region=paths_cfg.diffraction_lit_region,
                seed=self.cir_i,
            )
            if paths is None:
                # TODO: still need to write a CIR even if no paths are found.
                raise NotImplementedError("No paths available")

            paths_taps = paths.taps(
                bandwidth=paths_cfg.bandwidth,
                l_min=paths_cfg.l_min,
                l_max=paths_cfg.l_max,
                sampling_frequency=paths_cfg.sampling_frequency,
                num_time_steps=self.interpolation_factor,  # Doppler interpolation
                normalize=paths_cfg.normalize,
                normalize_delays=paths_cfg.normalize_delays,
                out_type="numpy",
            )

            # NOTE: if both the TX and RX are animated, we will get the outer
            # product of `parallelism` (all combinations of timesteps), even though
            # we only need them in sync.
            taps_results = prepare_and_normalize_cir_batch(
                taps=paths_taps,
                tx_index=np.arange(len(tx_instances)),
                rx_index=np.arange(len(rx_instances)),
                num_taps=self.num_taps,
                bandwidth=paths_cfg.bandwidth,
                snr_offset_db=paths_cfg.snr_offset_db,
                max_noise_std=self.main.cfg.srk_demo.max_noise_std,
            )

        taps_norm = taps_results["taps_norm"]
        taps = taps_results["taps"]
        assert taps_norm.dtype == np.float32
        assert taps.dtype == np.float32

        assert taps_norm.shape == (
            len(rx_instances),
            len(tx_instances),
            self.interpolation_factor,
        ), taps_norm.shape
        assert taps.shape == (
            len(rx_instances),
            len(tx_instances),
            self.interpolation_factor,
            self.num_taps * 2,
        ), taps.shape
        # Reshape (linearize) for convenience when writing out below.
        taps_norm = taps_norm.ravel()
        taps = taps.reshape(-1, self.num_taps * 2)

        # Strictly respect the CIR count even if not a multiple of the interpolation factor.
        # TODO: double-check this in the presence of parallelism
        if self.cir_i + taps_norm.shape[0] >= self.n_cirs:
            taps_norm = taps_norm[: self.n_cirs - self.cir_i]
            taps = taps[: self.n_cirs - self.cir_i, :]

        mode = "ab" if self.cir_i > 0 else "wb"
        with open(self.output_bin_fname, mode) as f:
            # For each CIR (including the interpolated ones), write:
            #     norm, tap0_real, tap0_imag, tap1_real, tap1_imag, ..., tapN_real, tapN_imag
            for k in range(taps.shape[0]):
                f.write(taps_norm[k].tobytes())
                f.write(taps[k, :].tobytes())
                self.n_cirs_written += 1

        self.cir_i += self.interpolation_factor * self.parallelism

        # Advance positions of our radio devices of interest along their trajectories
        # in preparation for the next iteration.
        for rd_type, props in self.devices.items():
            for rd_name, obj in props["instances"].items():
                if "trajectories" not in props:
                    continue
                traj = props["trajectories"][rd_name]

                traj.distance, traj.backward = traj.compute_next_distance(
                    traj.distance,
                    traj.backward,
                    self.time_delta * self.parallelism,
                    speed_multiplier=1.0,
                )
                obj.position, direction = traj.current_position_and_direction()
                obj.velocity = direction * traj.velocity
                dr.make_opaque(obj.position, obj.velocity)

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

    def _prepare_temporary_radio_devices(
        self, tx_index: int, rx_index: int
    ) -> tuple[dict[str, rt.Transmitter], dict[str, rt.Receiver]]:
        """
        Create temporary radio device objects based on the selected transmitter and receiver.
        One copy per `parallelism` value will be placed along the trajectory so that CIRs
        can be computed for all of them in parallel.
        """

        results = {}
        has_animation = False
        for rd_type, rd_index, source in (
            ("tx", tx_index, self.main.scene._transmitters),
            ("rx", rx_index, self.main.scene._receivers),
        ):
            name = list(source.keys())[rd_index]
            traj = self.main.animation_config.trajectories.get(name)

            results[rd_type] = {
                "index": rd_index,
                "name": name,
                "original_rd": source[name],
            }
            results[rd_type]["instances"] = {
                f"{rd_type}-tmp-batch-{i:04d}": copy(results[rd_type]["original_rd"])
                for i in range(self.parallelism if (traj is not None) else 1)
            }

            if traj is not None:
                if has_animation:
                    raise NotImplementedError(
                        "CIR batch export is not yet supported when both the TX and RX are animated."
                    )
                has_animation |= True

                # For each parallel instance, create a copy of the trajectory
                # offset by the time delta.
                results[rd_type]["trajectories"] = {}
                for instance_i, (k, obj) in enumerate(
                    results[rd_type]["instances"].items()
                ):
                    traj = deepcopy(traj)
                    traj.distance, traj.backward = traj.compute_next_distance(
                        traj.distance,
                        traj.backward,
                        self.time_delta * instance_i,
                        speed_multiplier=1.0,
                    )
                    obj.position, direction = traj.current_position_and_direction()
                    obj.velocity = direction * traj.velocity
                    dr.make_opaque(obj.position, obj.velocity)

                    results[rd_type]["trajectories"][k] = traj

        return results


@contextmanager
def replace_radio_devices(
    scene: rt.Scene, new_tx: dict[str, rt.Transmitter], new_rx: dict[str, rt.Receiver]
):
    tx_bak = scene._transmitters.copy()
    rx_bak = scene._receivers.copy()

    scene._transmitters = new_tx
    scene._receivers = new_rx

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
    parallelism: int,
) -> CirBatchExporter | None:
    if tx_index >= len(gui.scene._transmitters) or rx_index >= len(
        gui.scene._receivers
    ):
        return None

    # We will advance animations manually.
    gui.set_all_animations_playing(False)

    # TODO: auto-cancel if there's no animation.
    return CirBatchExporter(
        gui,
        tx_index,
        rx_index,
        output_filename,
        duration_s,
        sampling_frequency_hz,
        interpolation_factor,
        parallelism=parallelism,
    )
