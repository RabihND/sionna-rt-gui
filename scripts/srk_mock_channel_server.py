#!/usr/bin/env python3
"""
ZeroMQ server for sending configuration and receiving CIR data.
"""
import argparse
import logging

import numpy as np
import zmq

from common import add_project_root_to_path

add_project_root_to_path()

from sionna_rt_gui.srk_demo import SrkDemoConfig


class ChannelEmulatorServer:
    """
    Description of messages

    Sent by the client to request the config:
    msg_config_req = {
        "msg_type" : "config_req",
    }

    Sent by the server to respond to the config request:
    msg_config = {
        "msg_type" : "config_res",
        "num_taps" : 10, #int
        "fft_size" : 512, #int
        "subcarrier_spacing" : 30e3, #float
        "frequency" : 30e9 #float
    }

    Sent by the client to sent the CIR to the server:
    msg_cir = {
        "msg_type" : "cir",
        "taps" : taps.tolist(), # [2*num_taps (re, im)], np.float32
        "tap_indices" : tap_indices.tolist(), # [num_taps], uint16
        "noise_std" : 3.0 #float
    }

    Sent by the server to acknowledge the CIR:
    msg_cir_ack = {
        "msg_type" : "cir_ack",
    }

    Sent by the client to enable / disable the neural receiver:
    msg_nrx = {
        "msg_type" : "nrx",
        "enabled" : True,
    }

    Sent by the server to acknowledge the neural receiver config:
    msg_nrx_ack = {
        "msg_type" : "nrx_ack",
        "enabled" : True,
    }
    """

    def __init__(self, port: int):
        self.log = logging.getLogger(__name__)

        # ZeroMQ context and socket
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Control variables
        self.running = False

    def start_server(self):
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            self.log.info(f"Channel emulator server started on port {self.port}")
            self.log.info("Press Ctrl+C to stop")
            self.running = True

            while self.running:
                try:
                    self.log.info("Waiting for request from a client...")
                    response = self._handle_request(self.socket.recv_json())
                    if response is not None:
                        self.socket.send_json(response)

                except zmq.error.ZMQError as e:
                    self.log.error(f"ZMQ Error: {e}")
                    break
                except Exception as e:
                    self.log.error(f"Error handling request: {e}")

        except KeyboardInterrupt:
            self.log.info("server shutdown requested")
        finally:
            self._cleanup()

    def _handle_request(self, request: dict) -> dict | None:
        msg_type = request.get("msg_type")
        if msg_type == "config_req":
            self.log.info("Handling config request")
            return {
                "msg_type": "config_res",
                "num_taps": 10,  # int
                "fft_size": 512,  # int
                "subcarrier_spacing": 30e3,  # float
                "frequency": 30e9,  # float
            }

        elif msg_type == "cir":
            taps = np.array(request["taps"])
            tap_indices = np.array(request["tap_indices"])
            noise_std = request["noise_std"]
            self.log.info(
                f"Handling received CIR with {taps.shape} taps, {tap_indices.shape} tap indices, {noise_std:.2f} noise std"
            )
            # Actually, nothing to do in this mock server.
            return {
                "msg_type": "cir_ack",
            }

        elif msg_type == "nrx":
            enabled = request["enabled"]
            self.log.info(
                f"Handling received neural receiver config (enabled = {enabled})"
            )
            # Actually, nothing to do in this mock server.
            return {
                "msg_type": "nrx_ack",
                "enabled": enabled,
            }

        else:
            return None

    def stop_server(self):
        """Stop the server."""
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        self.log.info("Cleaning up server resources...")
        self.running = False
        self.socket.close()
        self.context.term()
        self.log.info("Server stopped")


def main():
    """Main function to run the channel emulator mock server."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = SrkDemoConfig()

    parser = argparse.ArgumentParser(description="ZeroMQ channel emulator mock server")
    parser.add_argument(
        "--port",
        type=int,
        default=cfg.channel_port,
        help=f"Port to bind server to (default: {cfg.channel_port})",
    )

    args = parser.parse_args()

    server = ChannelEmulatorServer(
        port=args.port,
    )
    server.start_server()


if __name__ == "__main__":
    main()
