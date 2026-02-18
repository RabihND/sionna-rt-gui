from __future__ import annotations

from collections import defaultdict
import logging
import os
import time

import numpy as np
import pandas as pd
import polyscope as ps
import polyscope.imgui as psim
import polyscope.implot as psplot

from . import PROJECT_DIR
from .animation import LoopingMode, animation_tick
from .drjit_util import read_gpu_utilization
from .config import (
    NVIDIA_GREEN,
    NVIDIA_GREEN_DARK,
    NVIDIA_GREEN_DARKER,
    SrkDemoConfig,
    StatsPlottingMode,
    STATS_PLOTTING_MODE_NAMES,
)
from .sionna_utils import prepare_valid_taps
from .srk_batch_export import export_cir_batch, CirBatchExporter
from .srk_stats_client import (
    UEStatsSubscriber,
    STATS_FIELDS_NAMES,
    STATS_FIELDS_RANGES,
    ReceiveResult as StatsStatus,
    load_stats_baselines,
)
from .srk_channel_client import ChannelEmulatorClient


class SrkDemo:
    def __init__(self, main: "SionnaRtGui", cfg: SrkDemoConfig):
        self.main = main
        self.log = logging.getLogger(__name__)
        self.cfg = cfg
        self.gpu_utilization: float | None = None
        self.last_gpu_utilization_timestamp: float | None = None

        # --- GUI / Polyscope
        self.plots_windows_flags = (
            psim.ImGuiWindowFlags_NoTitleBar | psim.ImGuiWindowFlags_NoDecoration
        )
        self.plots_axes_flags = (
            psplot.ImPlotAxisFlags_NoSideSwitch | psplot.ImPlotAxisFlags_NoHighlight
        )
        self.plots_size = (510, 230)

        # --- Channel emulator
        self.channel_client = ChannelEmulatorClient(
            self.main.cfg.srk_demo, self.main.cfg.paths
        )

        # Keep running min/max of each axis of the CIR plot so that it
        # doesn't jump around too much with animated UEs.
        self.cir_plot_vlims = [(np.inf, -np.inf), (np.inf, -np.inf)]
        self.batch_cir_exporter: CirBatchExporter | None = None

        # --- Stats
        self.stats_client = UEStatsSubscriber()
        self.stats_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(
                columns=["ue_id", "is_fake"] + list(STATS_FIELDS_NAMES)
            )
        )
        self.stats_baselines: dict[str, np.ndarray] | None = None
        if self.cfg.stats_baselines_npy is not None:
            fname = os.path.realpath(
                os.path.join(PROJECT_DIR, self.cfg.stats_baselines_npy)
            )
            self.stats_baselines = load_stats_baselines(
                fname, self.cfg.stats_baselines_source
            )
            print(f"[i] Loaded stats baselines from {fname}")
        # This timestamp will be used to align the live stats with the baseline stats.
        # It will be reset when restarting the demo / animation.
        self.stats_baselines_reference_timestamp: float | None = None

        # Try connecting to the channel and stats servers.
        # Connection failure is handled, the user can try again by clicking the "Connect" button.
        if self.channel_client.connect(
            server_host=self.cfg.channel_host,
            server_port=self.cfg.channel_port,
        ):
            self.channel_client.request_config()
        else:
            self.log.error(
                f"Failed to connect to channel server ({self.cfg.channel_host}:{self.cfg.channel_port})."
                " Click the 'Connect' button to try again."
            )
        if not self.stats_client.connect(
            topic=self.cfg.stats_topic,
            server_host=self.cfg.stats_host,
            server_port=self.cfg.stats_port,
        ):
            self.log.error(
                f"Failed to connect to stats server ({self.cfg.stats_host}:{self.cfg.stats_port})."
                " Click the 'Connect' button to try again."
            )

    # ------------------------

    def reset_state(self):
        """
        Reset stats history, running vlims, etc.
        Connections are kept intact.
        """
        self.stats_history.clear()
        self.stats_baselines_reference_timestamp = None
        self.cir_plot_vlims = [(np.inf, -np.inf), (np.inf, -np.inf)]

    # ------------------------

    def setup_demo_scenario(self):
        from sionna.rt.antenna_pattern import (
            antenna_pattern_registry,
            polarization_registry,
        )

        assert self.cfg.n_ues in (
            0,
            1,
            2,
        ), "The demo scenario is only implemented for 0, 1 or 2 UEs."

        self.reset_state()
        self.cfg.cir_max_delay_to_plot_ns = 1500.0

        gui = self.main
        cfg = gui.cfg

        # Antenna arrays
        for c in (cfg.tx_array, cfg.rx_array):
            c.num_rows = 1
            c.num_cols = 1
            c.polarization_i = polarization_registry.list().index("V")
        cfg.tx_array.pattern_i = antenna_pattern_registry.list().index("tr38901")
        cfg.rx_array.pattern_i = antenna_pattern_registry.list().index("iso")
        gui.scene._tx_array = cfg.tx_array.create()
        gui.scene._rx_array = cfg.rx_array.create()

        # General simulation parameters
        for c in (cfg.paths, cfg.radio_map):
            c.samples_per_tx = 10000000
            c.max_depth = 5
            c.los = True
            c.specular_reflection = True
            c.refraction = True
            c.diffraction = True
            c.diffraction_lit_region = False
            c.edge_diffraction = True
            c.diffuse_reflection = False

        # Reset radio devices, etc
        gui.clear_radio_map()
        gui.clear_selection()
        gui.clear_radio_devices()
        gui.clear_paths()

        # Ensure we use the right scene
        if "washington_dc" not in cfg.scene_filename:
            if "washington_dc" not in gui.built_in_scene_names:
                raise ValueError(
                    f'Scene "washington_dc" not found. Place the scene under `data/scenes/`.'
                    f" Current built-in scenes are: {gui.built_in_scene_names}"
                )

            cfg.scene_filename = gui.built_in_scenes["washington_dc"]
            gui.load_scene(
                "data/scenes/washington_dc/washington_dc.xml",
                recenter_camera=False,
            )

        # Camera pose
        self.main.home_camera_to_world = np.array(
            [
                [5.5471003e-01, -8.3204406e-01, -3.2004149e-08, -8.1965729e01],
                [5.7973224e-01, 3.8649887e-01, 7.1730834e-01, 3.4181381e01],
                [-5.9683138e-01, -3.9789820e-01, 6.9675630e-01, -5.4985059e02],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        self.main.move_camera_home()

        # Add transmitter
        tx = gui.add_radio_device(
            [-54.172, -248.828, 63], is_transmitter=True, allow_auto_update=False
        )
        tx.look_at([-72.702, 156.706, 1.500])

        # Add receiver(s)
        if self.cfg.n_ues >= 1:
            positions = [
                (19.701, -114.505, 1.500),
                (19.795, 159.378, 1.500),
                (-157.190, 160.221, 1.500),
                (-158.221, -123.441, 1.500),
                (-66.793, -145.738, 1.500),
            ]
            rx = gui.add_radio_device(
                positions[0], is_transmitter=False, allow_auto_update=False
            )
            traj = gui.animation_config.trajectories[rx.name]
            for pos in positions:
                traj.add_point(pos)
            traj.add_point(positions[0])  # Return to the start
            traj.enabled = self.cfg.play_animations
            traj.looping_mode_i = LoopingMode.Repeat.value
            # Starting position along the path
            traj.set(
                (0.92 if self.cfg.n_ues == 1 else 0.75) * traj.total_distance(),
                backward=False,
            )

        if self.cfg.n_ues >= 2:
            positions = [
                (34.118, -186.824, 1.500),
                (205.270, -185.245, 1.500),
                (197.912, -82.481, 1.500),
                (41.981, -146.337, 1.500),
            ]
            rx = gui.add_radio_device(
                positions[0], is_transmitter=False, allow_auto_update=False
            )
            traj = gui.animation_config.trajectories[rx.name]
            for pos in positions:
                traj.add_point(pos)
            traj.add_point(positions[0])  # Return to the start
            traj.enabled = self.cfg.play_animations
            traj.looping_mode_i = LoopingMode.Repeat.value
            # Starting position along the path
            traj.set(
                0.20 * traj.total_distance(),
                backward=False,
            )

        gui.animation_config.playing = True
        gui.animation_config.speed_multiplier = 7.0

        # Place the UEs at their starting positions along the trajectories.
        # Note we need to do this regardless of whether animations are actually playing.
        # This will also trigger re-computation and drawing of the paths.
        cfg.paths.auto_update = True
        animation_tick(
            gui, time_delta=0.0, force=True, allow_watchpoint_callbacks=False
        )

        # Force redraw at the next frame
        gui.reset_accumulation_requested = True

    # ------------------------

    def tick(self):

        if self.batch_cir_exporter is not None:
            try:
                _ = next(self.batch_cir_exporter)
            except StopIteration:
                self.batch_cir_exporter = None
                self.log.info("CIR batch export completed.")

            # The export process prevents anything else from happening
            return

        # Send updated CIR
        if self.channel_client.is_connected():
            self.channel_client.tick()

            # Apply newly received config, if any
            config = self.channel_client.pop_received_config()
            if config is not None:
                self.process_channel_config(config)

            if (self.main.paths_taps is not None) and self.cfg.cir_send_updates:
                self.channel_client.maybe_send_cir(
                    self.main.paths_taps, self.main._last_paths_update_time
                )

        # Update received stats
        if self.stats_client.is_connected():
            # Receive and process all available stats messages. They might have
            # queued up since the last tick, especially if framerate is low and
            # stats sending frequency is high.
            while True:
                status, result = self.stats_client.try_receive()
                if status == StatsStatus.SUCCESS:
                    self.process_ue_stats(result)
                else:
                    break

    # ------------------------

    def process_channel_config(self, config: dict):
        if "num_taps" in config:
            self.main.cfg.paths.num_taps = config["num_taps"]
        if "fft_size" in config:
            self.main.cfg.paths.fft_size = config["fft_size"]
        if "subcarrier_spacing" in config:
            self.main.cfg.paths.subcarrier_spacing = config["subcarrier_spacing"]
        if "frequency" in config:
            self.main.cfg.paths.frequency = config["frequency"]
        self.log.info(f"Applied received channel config: {config}")

    def set_use_neural_receiver(self, v: bool):
        if v == self.cfg.use_neural_receiver:
            # Nothing to do
            return

        # Value changed, send a message to the channel emulator server.
        # The config will be applied once we received an acknowledgment from the server.
        self.channel_client.send_neural_receiver_config(v)

    # ------------------------

    def process_ue_stats(self, ue_stats: dict):
        # For each UE in the stats, add a row to the history
        for i, new_stats in enumerate(ue_stats["UE_stats"]):
            ue_id = new_stats["ue_id"]
            stats_i = self.stats_history[ue_id]

            series = pd.Series(new_stats, index=stats_i.columns)
            series["timestamp"] = ue_stats["timestamp"]
            series["is_fake"] = ue_stats.get("is_fake", False)
            stats_i = pd.concat([stats_i, series.to_frame().T], ignore_index=True)

            # Limit the number of rows to the maximum number of entries per UE
            if len(stats_i) > self.cfg.stats_max_entries:
                stats_i = stats_i.iloc[-self.cfg.stats_max_entries :]

            # Sort by timestamp
            stats_i = stats_i.sort_values(by=["timestamp", "ue_id"])

            self.stats_history[ue_id] = stats_i

            # This is the first live stats entry we have received since
            # the animation / demo was started. Let's use its timestamp
            # to align the baseline stats with the live stats.
            if (
                (i == 0)
                and self.main.animation_config.playing
                and (self.stats_baselines_reference_timestamp is None)
            ):
                self.stats_baselines_reference_timestamp = ue_stats["timestamp"]

    # ------------------------

    def maybe_read_gpu_utilization(self) -> bool:
        if not self.cfg.show_gpu_utilization:
            return False

        now = time.time()
        should_refresh = (self.last_gpu_utilization_timestamp is None) or (
            now - self.last_gpu_utilization_timestamp
            >= self.cfg.gpu_utilization_interval_s
        )
        if not should_refresh:
            return False

        self.gpu_utilization = read_gpu_utilization()
        self.last_gpu_utilization_timestamp = now
        return True

    # ------------------------

    def gui(self):
        """
        Simplified demo GUI.
        """
        ui_scale = self.main.ui_scale

        # --- CIR batch export window
        if self.batch_cir_exporter is not None:
            self.batch_cir_exporter.gui()
            return

        # --- Controls window
        w = 375
        psim.SetNextWindowSize(
            (w * ui_scale, 270 * ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (10 * ui_scale, 10 * ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.Begin("Sionna Research Kit", open=True)

        bsize = (ui_scale * (w - 30) // 3, 30 * ui_scale)

        psim.NewLine()
        if self.channel_client.is_connected():
            # Prominent button for the NRX ON / OFF button
            has_nrx_req_pending = self.channel_client.has_pending_nrx_request()
            highlight_button = not has_nrx_req_pending and self.cfg.use_neural_receiver
            psim.BeginDisabled(has_nrx_req_pending)
            if highlight_button:
                psim.PushStyleColor(psim.ImGuiCol_Text, (0, 0, 0, 1))
                psim.PushStyleColor(psim.ImGuiCol_Button, NVIDIA_GREEN)
                psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, NVIDIA_GREEN_DARK)
                psim.PushStyleColor(psim.ImGuiCol_ButtonActive, NVIDIA_GREEN_DARKER)

            psim.SetCursorPosX((0.5 * bsize[0] + 15 * ui_scale))
            label = "NRX ON" if self.cfg.use_neural_receiver else "NRX OFF"
            if psim.Button(label, size=(2 * bsize[0], bsize[1])):
                self.set_use_neural_receiver(not self.cfg.use_neural_receiver)

            if highlight_button:
                psim.PopStyleColor()
                psim.PopStyleColor()
                psim.PopStyleColor()
                psim.PopStyleColor()
            psim.EndDisabled()
            psim.NewLine()

        if self.cfg.show_gpu_utilization:
            self.maybe_read_gpu_utilization()
            psim.Spacing()
            txt = f"GPU utilization: {self.gpu_utilization}%"
            w_scaled, _ = psim.CalcTextSize(txt)
            psim.SetCursorPosX(0.5 * (bsize[0] + w_scaled))
            psim.Text(txt)

        if psim.Button("Play##srk", size=bsize):
            self.main.set_all_animations_playing(True)
        psim.SameLine()
        if psim.Button("Pause##srk", size=bsize):
            self.main.set_all_animations_playing(False)
        psim.SameLine()
        if psim.Button("Reset##srk", size=bsize):
            self.setup_demo_scenario()

        psim.NewLine()

        # - Channel emulator server
        if psim.TreeNode("Channel emulator##srk"):
            psim.Spacing()
            is_connected = self.channel_client.is_connected()

            # TODO: controls for host & port
            if is_connected:
                if self.channel_client.has_pending_config_request():
                    psim.BeginDisabled(True)
                    psim.Button("Config pending...")
                    psim.EndDisabled()
                elif psim.Button("Request config##channel"):
                    self.channel_client.request_config()

                psim.SameLine()
                psim.BeginDisabled(self.main.paths_taps is None)
                if psim.Button("Send CIR##channel"):
                    # Note: we bypass `cfg.cir_send_updates` since the user requested sending
                    # the CIR explicitly.
                    self.channel_client.maybe_send_cir(
                        self.main.paths_taps,
                        time.time(),
                        skip_throttle=True,
                    )
                psim.EndDisabled()

                psim.Spacing()
                _, self.cfg.cir_send_updates = psim.Checkbox(
                    "Send CIRs to channel emulator",
                    self.cfg.cir_send_updates,
                )

                psim.Spacing()
                psim.Text(
                    f"Connected to channel server at {self.cfg.channel_host}:{self.cfg.channel_port}."
                )
                if psim.Button("Disconnect##channel"):
                    self.channel_client.disconnect()
            else:
                psim.Text(
                    f"Not connected to channel server at {self.cfg.channel_host}:{self.cfg.channel_port}."
                )
                if psim.Button("Connect##channel"):
                    self.channel_client.connect(
                        topic=None,
                        server_host=self.cfg.channel_host,
                        server_port=self.cfg.channel_port,
                    )
            psim.NewLine()

            psim.SetNextItemWidth(185 * ui_scale)
            changed_offset, self.main.cfg.paths.snr_offset_db = psim.InputFloat(
                "SNR offset (dB)",
                self.main.cfg.paths.snr_offset_db,
                format="%.1f",
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue,
            )

            psim.SetNextItemWidth(185 * ui_scale)
            changed_max_noise, self.cfg.max_noise_std_db = psim.InputFloat(
                "Max noise std (dB)",
                self.cfg.max_noise_std_db,
                format="%.2f",
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue,
            )

            if changed_offset or changed_max_noise:
                # Trigger re-send of the CIR
                self.main._last_paths_update_time = time.time()

            # - CIR batch export
            _, self.cfg.cir_export_tx_index = psim.Combo(
                "From TX",
                self.cfg.cir_export_tx_index,
                items=[name for name in self.main.scene._transmitters.keys()],
            )
            _, self.cfg.cir_export_rx_index = psim.Combo(
                "To RX",
                self.cfg.cir_export_rx_index,
                items=[name for name in self.main.scene._receivers.keys()],
            )

            psim.SetNextItemWidth(185 * ui_scale)
            _, self.cfg.cir_output_filename = psim.InputText(
                "Output filename", self.cfg.cir_output_filename
            )
            psim.SetNextItemWidth(185 * ui_scale)
            _, self.cfg.cir_export_duration_s = psim.InputFloat(
                "Export duration (s)", self.cfg.cir_export_duration_s
            )

            if psim.Button("CIR batch export##cir_batch_export"):
                sampling_frequency_hz = (
                    self.cfg.cir_sampling_frequency_hz
                    if self.cfg.cir_sampling_frequency_hz is not None
                    else self.main.cfg.paths.subcarrier_spacing
                )
                self.batch_cir_exporter = export_cir_batch(
                    gui=self.main,
                    tx_index=self.cfg.cir_export_tx_index,
                    rx_index=self.cfg.cir_export_rx_index,
                    output_filename=self.cfg.cir_output_filename,
                    duration_s=self.cfg.cir_export_duration_s,
                    sampling_frequency_hz=sampling_frequency_hz,
                    interpolation_factor=self.cfg.cir_export_interpolation_factor,
                    parallelism=self.cfg.cir_export_parallelism,
                )

            psim.NewLine()

            psim.TreePop()

        psim.Spacing()

        # - Stats server
        if psim.TreeNode("Stats##srk"):
            psim.Spacing()
            is_connected = self.stats_client.is_connected()

            psim.SetNextItemWidth(185 * ui_scale)
            _, self.cfg.stats_plotting_mode_i = psim.Combo(
                "Plot stats##stats_plotting_mode",
                self.cfg.stats_plotting_mode_i,
                items=STATS_PLOTTING_MODE_NAMES,
            )

            psim.Spacing()
            if psim.Button("Clear stats history"):
                self.stats_history.clear()

            # TODO: controls for host & port
            psim.Spacing()
            if is_connected:
                psim.Text(
                    f"Connected to stats server at {self.cfg.stats_host}:{self.cfg.stats_port}."
                )
                if psim.Button("Disconnect##stats"):
                    self.stats_client.disconnect()
            else:
                psim.Text(
                    f"Not connected to stats server at {self.cfg.stats_host}:{self.cfg.stats_port}."
                )
                if psim.Button("Connect##stats"):
                    self.stats_client.connect(
                        topic=self.cfg.stats_topic,
                        server_host=self.cfg.stats_host,
                        server_port=self.cfg.stats_port,
                    )

            psim.NewLine()
            psim.TreePop()

        psim.End()

        # --- CIR window
        taps: np.ndarray | None = self.main.paths_taps
        if self.cfg.show_cir and (taps is not None):
            # Place window in the top-right corner of the screen
            window_resolution = ps.get_window_size()
            w = self.plots_size[0]
            psim.SetNextWindowSize((w * ui_scale, self.plots_size[1] * ui_scale))
            psim.SetNextWindowPos(
                (window_resolution[0] - (w + 10) * ui_scale, 10 * ui_scale)
            )

            psim.Begin(
                "CIR plot",
                open=True,
                flags=self.plots_windows_flags,
            )
            psim.SetCursorPosX(0)
            psim.SetCursorPosY(0)

            psplot.BeginPlot(
                "Channel impulse response",
                size=(self.plots_size[0] * ui_scale, self.plots_size[1] * ui_scale),
            )
            psplot.SetupLegend(
                psplot.ImPlotLocation_NorthEast,
                flags=psplot.ImPlotLegendFlags_Horizontal,
            )
            psplot.SetupAxis(
                psplot.ImAxis_X1,
                label="Delay (ns)",
                flags=self.plots_axes_flags,
            )
            psplot.SetupAxis(
                psplot.ImAxis_Y1,
                label="|a|",
                flags=self.plots_axes_flags,
            )
            psplot.SetupAxisFormat(
                psplot.ImAxis_Y1,
                fmt="%0.1e",
            )

            # TODO: let the user decide which tx to plot taps for?
            tx_idx_for_cir = 0
            a, tau = self.main.paths_cir
            # Two lists of valid tau, a.
            a_abs, tau, n_valid, ylims, xlims = prepare_valid_taps(
                a, tau, tx_idx_for_cir, return_vlims=True
            )
            # Update running vlims
            for i, lims in enumerate((xlims, ylims)):
                self.cir_plot_vlims[i] = (
                    min(lims[0], self.cir_plot_vlims[i][0]),
                    max(lims[1], self.cir_plot_vlims[i][1]),
                )
            if self.cfg.cir_max_delay_to_plot_ns is not None:
                self.cir_plot_vlims[0] = (
                    0,
                    min(self.cir_plot_vlims[0][1], self.cfg.cir_max_delay_to_plot_ns),
                )
            xlims, ylims = self.cir_plot_vlims

            if n_valid > 0:
                # This can happen easily if there's a single path.
                if xlims[0] == xlims[1]:
                    xlims = (xlims[0], xlims[1] + 1)
                if ylims[0] == ylims[1]:
                    vlog = np.floor(np.log10(ylims[0]))
                    ylims = (np.power(10, vlog), np.power(10, vlog + 1))
            else:
                xlims = (0, 1)
                ylims = (0, 1)

            ex = xlims[1] - xlims[0]
            ey = ylims[1] - ylims[0]
            psplot.SetupAxesLimits(
                x_min=xlims[0] - 0.05 * ex,
                x_max=xlims[1] + 0.05 * ex,
                y_min=ylims[0] - 0.1 * ey,
                y_max=ylims[1] + 0.1 * ey,
                cond=psplot.ImPlotCond_Always,
            )

            for rx_idx, (valid_a, valid_tau) in enumerate(zip(a_abs, tau)):
                # Note: the interface takes fp64 and there's some weird rounding issues
                # when passing fp32 NumPy arrays.
                psplot.PlotScatter(
                    f"UE {rx_idx:d}",
                    valid_tau.astype(np.float64),
                    valid_a.astype(np.float64),
                )

            psplot.EndPlot()
            psim.End()

        # --- Stats window
        if self.stats_history and (
            self.cfg.stats_plotting_mode_i != StatsPlottingMode.DISABLED.value
        ):

            n_plots = len(self.cfg.stats_to_plot)
            directions = {
                StatsPlottingMode.DOWNLINK.value: [("down", "DL")],
                StatsPlottingMode.UPLINK.value: [("up", "UL")],
                StatsPlottingMode.BOTH.value: [("down", "DL"), ("up", "UL")],
            }[self.cfg.stats_plotting_mode_i]

            # Place window in the bottom-right corner of the screen
            window_resolution = ps.get_window_size()
            w = self.plots_size[0]
            h = n_plots * self.plots_size[1]
            psim.SetNextWindowSize((w * ui_scale, h * ui_scale))
            psim.SetNextWindowPos(
                (
                    window_resolution[0] - (w + 10) * ui_scale,
                    window_resolution[1] - (h + 10) * ui_scale,
                )
            )

            psim.Begin(
                "UE Stats",
                open=True,
                flags=self.plots_windows_flags,
            )
            psim.SetCursorPosX(0)
            psim.SetCursorPosY(0)

            has_fake_data = False
            ue_ids = sorted(self.stats_history.keys())
            for selected_ue in ue_ids:
                has_fake_data |= self.stats_history[selected_ue]["is_fake"].any()

            plot_title = (
                f"UE {STATS_PLOTTING_MODE_NAMES[self.cfg.stats_plotting_mode_i]} stats"
            )
            psplot.BeginSubplots(
                plot_title + (" (fake data)" if has_fake_data else ""),
                cols=1,
                rows=n_plots,
                size=(
                    self.plots_size[0] * ui_scale,
                    n_plots * self.plots_size[1] * ui_scale,
                ),
                flags=psplot.ImPlotSubplotFlags_LinkCols,
            )

            for plot_i, field_name in enumerate(self.cfg.stats_to_plot):

                # Show legend only for the first plot
                plot_flags = 0 if (plot_i == 0) else psplot.ImPlotFlags_NoLegend
                psplot.BeginPlot("", flags=plot_flags)
                if plot_i == 0:
                    psplot.SetupLegend(
                        psplot.ImPlotLocation_NorthEast,
                        flags=psplot.ImPlotLegendFlags_Horizontal,
                    )
                psplot.SetupAxis(
                    psplot.ImAxis_Y1,
                    label=STATS_FIELDS_NAMES[field_name],
                    flags=self.plots_axes_flags,
                )

                vmin, vmax = np.inf, -np.inf
                for direction, _ in directions:
                    vmin_d, vmax_d = STATS_FIELDS_RANGES[f"{field_name}_{direction}"]
                    vmin = min(vmin, vmin_d)
                    vmax = max(vmax, vmax_d)

                vpad = 0.05 * (vmax - vmin)
                psplot.SetupAxesLimits(
                    # Make sure all entries are visible
                    x_min=-self.cfg.stats_max_entries / self.cfg.stats_frequency_hz,
                    x_max=0,
                    y_min=vmin - vpad,
                    y_max=vmax + vpad,
                )

                show_x_label = plot_i == len(self.cfg.stats_to_plot) - 1
                psplot.SetupAxis(
                    psplot.ImAxis_X1,
                    label="Time (s, relative)" if show_x_label else "",
                    flags=self.plots_axes_flags
                    | (psplot.ImPlotAxisFlags_NoTickLabels if not show_x_label else 0),
                )

                # TODO: better color choice or different line styles
                call_i = 0
                for selected_ue in ue_ids:
                    for direction, dir_label in directions:
                        if call_i == 0:
                            psplot.PushStyleColor(
                                psplot.ImPlotCol_MarkerFill, NVIDIA_GREEN_DARK
                            )
                            psplot.PushStyleColor(
                                psplot.ImPlotCol_Line, NVIDIA_GREEN_DARK
                            )
                            psplot.PushStyleVar(
                                psplot.ImPlotStyleVar_LineWeight, 3.0 * ui_scale
                            )

                        ue_stats = self.stats_history[selected_ue]
                        timestamps = (
                            ue_stats["timestamp"] - ue_stats["timestamp"].max()
                        ) / 1000
                        psplot.PlotLine(
                            f"{selected_ue} ({dir_label})",
                            timestamps.values.astype(np.float64),
                            ue_stats[f"{field_name}_{direction}"].values.astype(
                                np.float64
                            ),
                        )

                        if call_i == 0:
                            psplot.PopStyleColor()
                            psplot.PopStyleColor()
                            psplot.PopStyleVar()

                        call_i += 1

                if (self.stats_baselines is not None) and (
                    self.stats_baselines_reference_timestamp is not None
                ):
                    baseline_stats = self.stats_baselines[field_name]
                    timestamps = self.stats_baselines["timestamp"]
                    # Align the references' timestamps with the live stats
                    timestamps = timestamps - timestamps[0]
                    # Reuse the last UE's max timestamp to make time progress
                    current_time_ref = (
                        ue_stats["timestamp"].max()
                        - self.stats_baselines_reference_timestamp
                    )
                    timestamps = (timestamps - current_time_ref) / 1000
                    # TODO: preserve existing x axis limits for this call
                    psplot.PlotLine(
                        "Baseline",
                        timestamps.astype(np.float64),
                        baseline_stats.astype(np.float64),
                    )

                psplot.EndPlot()

            psplot.EndSubplots()

            psim.End()  # End stats window

        # --- OSM credit window
        if self.cfg.osm_credit_string is not None:
            window_resolution = ps.get_window_size()
            w_scaled, h_scaled = psim.CalcTextSize(self.cfg.osm_credit_string)
            w_scaled += 15 * ui_scale
            h_scaled += 10 * ui_scale
            psim.SetNextWindowSize((w_scaled, h_scaled))
            psim.SetNextWindowPos((0, window_resolution[1] - h_scaled))

            psim.Begin(
                "OSM credit",
                open=True,
                flags=self.plots_windows_flags,
            )

            psim.SetCursorPosX(5 * ui_scale)
            psim.SetCursorPosY(5 * ui_scale)
            psim.Text(self.cfg.osm_credit_string)

            psim.End()
