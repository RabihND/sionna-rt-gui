#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
ZeroMQ Client for receiving UE statistics from a server.
Subscribes to UE stats data published by the server at regular intervals.
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


if __name__ == "__main__":
    add_project_root_to_path()

from sionna_rt_gui.srk_demo import SrkDemoConfig
from sionna_rt_gui.srk_stats_client import UEStatsSubscriber, ReceiveResult


class UEStatsMockClient(UEStatsSubscriber):

    def __init__(self, sleep_delay_s: float = 0.1):
        """
        Initialize the ZeroMQ UE stats subscriber.

        Args:
            sleep_delay_s: How long to sleep between queue polls, in seconds.
        """
        super().__init__()
        # Control variables
        self.running = False
        self.sleep_delay_s = sleep_delay_s

    def start_subscriber(self, max_messages: Optional[int] = None):
        """
        Start subscribing to UE stats data (when running as a standalone client script).

        Args:
            max_messages: Maximum number of messages to receive (None for unlimited)
        """
        try:
            self.log.info("Starting UE stats subscriber...")
            self.log.info(
                f"Waiting for UE stats data (max_messages: {max_messages or 'unlimited'})"
            )
            self.log.info("Press Ctrl+C to stop")

            self.running = True
            received_count = 0

            while self.running:
                status, result = self.try_receive()

                if status == ReceiveResult.SUCCESS:
                    assert isinstance(result, dict)
                    self._process_ue_stats(result)
                    received_count += 1

                    if max_messages and received_count >= max_messages:
                        self.log.info(f"Reached maximum message limit: {max_messages}")
                        break
                elif status == ReceiveResult.NO_MESSAGE:
                    time.sleep(self.sleep_delay_s)
                elif status == ReceiveResult.DECODE_ERROR:
                    self.log.error(f"Failed to decode JSON message:\n{repr(result)}")
                    time.sleep(10 * self.sleep_delay_s)
                elif status == ReceiveResult.OTHER_ERROR:
                    self.log.error(f"Error receiving message:\n{repr(result)}")
                    time.sleep(10 * self.sleep_delay_s)

        except KeyboardInterrupt:
            self.log.info("Subscriber shutdown requested")

    def stop_subscriber(self):
        """Stop the subscriber."""
        self.running = False
        self.log.info("Stopping subscriber...")


def stats_demo(cfg: SrkDemoConfig, max_messages: int, quiet: bool):
    logger = logging.getLogger(__name__)
    # Configure logging level
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create subscriber
    subscriber = UEStatsMockClient()

    try:
        # Connect to publisher
        if not subscriber.connect(
            topic=cfg.stats_topic,
            server_host=cfg.stats_host,
            server_port=cfg.stats_port,
        ):
            logger.error(
                f"Failed to connect to publisher ({cfg.stats_host}:{cfg.stats_port})"
            )
            return

        # Start subscribing
        subscriber.start_subscriber(max_messages=max_messages)

        # Print final statistics
        stats = subscriber.get_statistics()
        logger.info(f"Final statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Subscription interrupted by user")
    except Exception as e:
        # TODO: if queue is full, maybe wait a while and try again.
        logger.error(f"Unexpected error: {e}")


def main():
    """Main function to run the UE stats subscriber."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = SrkDemoConfig()

    parser = argparse.ArgumentParser(description="ZeroMQ UE Statistics Subscriber")
    parser.add_argument(
        "--host",
        type=str,
        default=cfg.stats_host,
        help=f"Publisher hostname or IP (default: {cfg.stats_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=cfg.stats_port,
        help=f"Publisher port (default: {cfg.stats_port})",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=cfg.stats_topic,
        help=f"Topic to subscribe to (default: {cfg.stats_topic})",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Maximum number of messages to receive (default: unlimited)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()
    cfg.stats_host = args.host
    cfg.stats_port = args.port
    cfg.stats_topic = args.topic
    del args.host, args.port, args.topic

    stats_demo(cfg, **vars(args))


if __name__ == "__main__":
    main()
