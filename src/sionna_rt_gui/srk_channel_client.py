from __future__ import annotations

import time

import numpy as np
import zmq

from .config import PathsConfig, SrkDemoConfig
from .sionna_utils import prepare_and_normalize_cir
from .srk_client_base import SrkClientBase, ReceiveResult


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
        # Maps the message type to (message, timestamp, skip_throttle).
        self._pending_messages: dict[str, tuple[dict, float, bool]] = {}

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
            for message_type, (
                message,
                timestamp,
                skip_throttle,
            ) in self._pending_messages.items():
                # Throttling logic to avoid flooding the server with CIR messages when
                # a UE is being moved at each frame.
                if message_type == "cir" and not skip_throttle:
                    time_since_cir_update = (
                        time.time() - self._sent_message_timestamps.get(message_type, 0)
                    )
                    if time_since_cir_update <= self.srk_cfg.cir_min_send_delay_s:
                        self.log.debug(
                            f"Throttling CIR message because {time_since_cir_update=:.2f}s < minimum delay {self.srk_cfg.cir_min_send_delay_s:.2f}s"
                        )
                        continue

                self.socket.send_json(message)
                if message_type == "cir":
                    self.log.info("Sent CIR message to channel emulator server.")
                else:
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
                self.log.info(
                    "Received CIR acknowledgment from channel emulator server."
                )

            elif msg_type == "nrx_ack":
                self.log.debug("Received neural receiver config acknowledgment")
                self.srk_cfg.use_neural_receiver = result.get("enabled", False)

            elif msg_type == "error":
                self.log.error(
                    "Channel emulator server error: %s (details: %s)",
                    result.get("error", "unknown"),
                    result.get("details", "none"),
                )

            self._response_pending_since = None
            self._response_pending_message_type = None

        return status, result

    def queue_message(
        self, message: dict, timestamp: float, skip_throttle: bool = False
    ) -> bool:
        # Note: the previous pending message of this type, if any, will be overwritten.
        self._pending_messages[message["msg_type"]] = (
            message,
            timestamp,
            skip_throttle,
        )

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

    def send_neural_receiver_config(self, use_neural_receiver: bool) -> bool:
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

        message = prepare_and_normalize_cir(
            taps,
            tx_index=0,
            rx_index=0,
            num_taps=self.paths_cfg.num_taps,
            snr_offset_db=self.paths_cfg.snr_offset_db,
            max_noise_std=self.srk_cfg.max_noise_std,
        )
        # Only one CIR can be pending at a time (overwrites previous). Log only when
        # we're adding a new pending CIR, not every time we overwrite it while waiting for ack.
        was_pending = "cir" in self._pending_messages
        self.queue_message(message, last_changed_timestamp, skip_throttle=skip_throttle)
        if not was_pending:
            self.log.info(
                "Queued CIR message (norms=%d, taps=%d, tap_indices=%d)",
                len(message["norms"]),
                len(message["taps"]),
                len(message["tap_indices"]),
            )
        return True
