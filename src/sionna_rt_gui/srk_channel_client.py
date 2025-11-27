from __future__ import annotations

import time

import numpy as np
import zmq

from .config import PathsConfig, SrkDemoConfig
from .srk_client_base import SrkClientBase, ReceiveResult

# Thermal noise power helper (kTB)
K_BOLTZMANN = 1.380649e-23  # J/K


def compute_thermal_noise_power(
    bandwidth_hz: float, temperature_k: float = 290.0
) -> float:
    """Compute thermal noise power in Watts for given bandwidth at temperature."""
    return K_BOLTZMANN * temperature_k * bandwidth_hz


class ChannelEmulatorClient(SrkClientBase):
    """
    Helper class to receive configurations and send CIR data to a ZeroMQ server.
    """

    def __init__(
        self,
        srk_cfg: SrkDemoConfig,
        paths_cfg: PathsConfig,
        ack_wait_timeout_s: float = 5.0,
    ):
        super().__init__("channel emulator", zmq.REQ)

        self.srk_cfg = srk_cfg
        self.paths_cfg = paths_cfg
        self.ack_wait_timeout_s = ack_wait_timeout_s

        # Latest received config, if any.
        self._latest_config: dict | None = None
        # Timestamp of the last message we sent of this type.
        # Used to avoid sending the same message multiple times.
        # Note that the timestamp may correspond to the time the data was produced,
        # rather than the time the message was actually sent.
        self._sent_message_timestamps: dict[str, float] = {}

        # Whether we are waiting for a response to the last message we sent.
        self._response_pending_since: float | None = None
        self._response_pending_message_type: str | None = None
        # Since we cannot send new messages until the last one is acknowledged,
        # we queue at most one message of each type.
        # Maps the message type to (message, timestamp).
        self._pending_messages: dict[str, tuple[dict, float]] = {}

    # ------------------------

    def tick(self):
        if not self.is_connected():
            return

        # Check if we have received any ACKs or responses.
        self.try_receive()

        # We must wait for the response to the last message we sent.
        if self._response_pending_since is not None:
            # TODO: if we have been waiting for too long, probably the
            # server wasn't listening on the other side or the message was dropped,
            # so we should forget it to avoid blocking future messages.
            elapsed = time.time() - self._response_pending_since
            if elapsed > self.ack_wait_timeout_s:
                self.log.warning(
                    f"Message of type '{self._response_pending_message_type}' was not"
                    f" acknowledged after {self.ack_wait_timeout_s}s, resetting connection."
                )
                self._response_pending_since = None
                self._response_pending_message_type = None
                self.reconnect()

        else:
            # Send the first pending message, if any. Other queued messages
            # will have to keep waiting.
            to_delete = None
            for message_type, (message, timestamp) in self._pending_messages.items():
                # Throttling logic to avoid flooding the server with CIR messages when
                # a UE is being moved at each frame.
                if message_type == "cir" and not message.get("skip_throttle", False):
                    time_since_cir_update = (
                        time.time() - self._sent_message_timestamps.get(message_type, 0)
                    )
                    if time_since_cir_update <= self.srk_cfg.cir_min_send_delay_s:
                        self.log.debug(
                            f"Throttling CIR message because {time_since_cir_update=:.2f}s < minimum delay {self.srk_cfg.cir_min_send_delay_s:.2f}s"
                        )
                        continue

                self.socket.send_json(message)
                self.log.debug("Sent '%s' message", message_type)
                self._sent_message_timestamps[message_type] = timestamp
                self._response_pending_since = time.time()
                self._response_pending_message_type = message_type
                to_delete = message_type
                break
            if to_delete is not None:
                del self._pending_messages[to_delete]

    def try_receive(self) -> tuple[ReceiveResult, dict | Exception | None]:
        status, result = super().try_receive()
        if status == ReceiveResult.SUCCESS:
            msg_type = result.get("msg_type")
            if msg_type == "config_res":
                self.log.debug("Received config from channel emulator server")
                self._latest_config = result

                if "cir" in self._pending_messages:
                    self.log.debug(
                        "Dropped pending CIR message because a new config was received."
                    )
                    del self._pending_messages["cir"]

            elif msg_type == "cir_ack":
                self.log.debug("Received CIR acknowledgment")

            elif msg_type == "nrx_ack":
                self.log.debug("Received neural receiver config acknowledgment")
                self.srk_cfg.use_neural_receiver = result.get("enabled", False)

            self._response_pending_since = None
            self._response_pending_message_type = None

        return status, result

    def queue_message(self, message: dict, timestamp: float) -> bool:
        # Note: the previous pending message of this type, if any, will be overwritten.
        self._pending_messages[message["msg_type"]] = (message, timestamp)

    # ------------------------

    def request_config(self) -> bool:
        if not self.is_connected():
            self.log.debug(
                "Not connected to channel emulator server, cannot send config request."
            )
            return False

        self.queue_message({"msg_type": "config_req"}, time.time())
        self.log.info("Requested config from channel emulator server.")
        return True

    def has_pending_config_request(self) -> bool:
        return (self._response_pending_message_type == "config_req") or (
            "config_req" in self._pending_messages
        )

    def pop_received_config(self) -> dict | None:
        result = self._latest_config
        self._latest_config = None
        return result

    # ------------------------

    def send_neural_receiver_config(self, use_neural_receiver: bool):
        if not self.is_connected():
            self.log.debug(
                "Not connected to channel emulator server, cannot send neural receiver config."
            )
            return False

        self.queue_message(
            {"msg_type": "nrx", "enabled": use_neural_receiver}, time.time()
        )
        self.log.debug(
            f"Queued neural receiver message (enabled = {use_neural_receiver}) to channel emulator server."
        )
        return True

    def has_pending_nrx_request(self) -> bool:
        return (self._response_pending_message_type == "nrx") or (
            "nrx" in self._pending_messages
        )

    # ------------------------

    def maybe_send_cir(
        self,
        taps: np.ndarray | None,
        last_changed_timestamp: float,
        skip_throttle: bool = False,
    ) -> bool:
        if not self.is_connected():
            self.log.debug("Not connected to channel emulator server, cannot send CIR.")
            return False

        if taps is None:
            return False

        last_sent = self._sent_message_timestamps.get("cir")
        if (last_sent is not None) and (last_changed_timestamp <= last_sent):
            # self.log.debug("Skipping CIR send because it's the same as the last one")
            return False

        message = self._prepare_cir(taps)
        if skip_throttle:
            message["skip_throttle"] = True
        self.queue_message(message, last_changed_timestamp)
        return True

    def _prepare_cir(self, taps: np.ndarray) -> dict:
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        # TODO: should we make the tx/rx selection configurable?
        taps = taps[0, 0, 0, 0, 0, ...]
        # Retain only the num_taps largest absolute taps
        tap_indices = np.argsort(np.abs(taps))[::-1][: self.paths_cfg.num_taps]
        taps = taps[tap_indices]

        # Convert to real/imag compatible with the C complex type
        taps = np.reshape(np.stack([taps.real, taps.imag], axis=1), [-1])

        # Thermal noise power (kTB) in Watts at 290 K
        thermal_noise_power_w = compute_thermal_noise_power(self.paths_cfg.bandwidth)
        noise_std = float(np.sqrt(thermal_noise_power_w))

        # Normalize taps and compute scaling norm
        norm = np.sqrt(np.sum(taps**2))
        if norm == 0:
            noise_std = 0
        else:
            taps /= norm
            noise_std /= norm

        snr_offset_factor = 10 ** (-self.paths_cfg.snr_offset_db / 20.0)
        noise_std *= snr_offset_factor

        taps = taps.astype(np.float32)
        tap_indices = tap_indices.astype(np.uint16)
        noise_std = float(min(noise_std, self.srk_cfg.max_noise_std))
        return {
            "msg_type": "cir",
            "taps": taps.tolist(),
            "tap_indices": tap_indices.tolist(),
            "taps_norm": float(norm),
            "noise_std": noise_std,
        }
