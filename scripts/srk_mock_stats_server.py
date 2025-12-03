#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
ZeroMQ server for sending UE statistics to clients at regular intervals.
Publishes UE stats data every second to subscribed clients.
"""
import argparse
import logging
import os
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


if __name__ == "__main__":
    add_project_root_to_path()

from sionna_rt_gui import DATA_DIR
from sionna_rt_gui.srk_demo import SrkDemoConfig
from sionna_rt_gui.srk_stats_server import UEStatsPublisher


def main():
    """Main function to run the UE stats publisher."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = SrkDemoConfig()
    data_dir = os.path.join(DATA_DIR, "srk")

    parser = argparse.ArgumentParser(
        description="ZeroMQ UE statistics publisher mock server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=cfg.stats_port,
        help=f"Port to bind publisher to (default: {cfg.stats_port})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Publish interval in seconds (default: 0.2 seconds)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=data_dir,
        help=f"Directory containing sample data files (default: {data_dir})",
    )

    args = parser.parse_args()

    publisher = UEStatsPublisher(
        topic=cfg.stats_topic,
        port=args.port,
        interval=args.interval,
        data_dir=args.data_dir,
        n_ues=cfg.n_ues,
    )
    publisher.start_publisher()


if __name__ == "__main__":
    main()
