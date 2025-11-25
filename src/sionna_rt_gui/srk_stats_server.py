import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import zmq


from .srk_stats_client import STATS_FIELDS_TYPES, STATS_FIELDS_RANGES


class UEStatsPublisher:
    def __init__(
        self,
        topic: str,
        port: int,
        interval: float = 1.0,
        data_dir: str = ".",
        n_ues: int = 1,
    ):
        """
        Initialize the ZeroMQ UE stats publisher.

        Args:
            port: Port to bind the publisher to
            interval: Interval in seconds to send data
            data_dir: Directory containing sample JSON files
        """
        self.log = logging.getLogger(__name__)

        self.topic = topic
        self.port = port
        self.interval = interval
        self.data_dir = Path(data_dir)
        self.n_ues = n_ues

        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)  # Publisher socket

        # Load sample data files
        self.rng = np.random.default_rng(1234)
        self.curent_data: dict[str, Any] = self._load_sample_data()

        # Control variables
        self.running = False

    def _load_sample_data(self):
        """Load sample JSON data files."""
        json_files = list(self.data_dir.glob("sample_data*.json"))
        available = []
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    available.append(data)
                self.log.info(f"Loaded sample data from {file_path.name}")
            except Exception as e:
                self.log.error(f"Failed to load {file_path}: {e}")

        if not available:
            self.log.error("No sample data files found!")
            raise FileNotFoundError(
                "No sample_data*.json files found in data directory"
            )

        self.log.info(f"Loaded {len(available)} sample data files")

        chosen = self.rng.choice(available).copy()
        # Only keep the first n_ues UEs
        chosen["UE_stats"] = chosen["UE_stats"][: self.n_ues]
        return chosen

    def _generate_live_ue_stats(self) -> Dict[str, Any]:
        """Generate live UE stats from sample data."""
        # Update the current data with some kind of random walk
        data = self.curent_data

        data["timestamp"] = int(time.time() * 1000)
        for ue in data["UE_stats"]:
            for k, v in ue.items():
                if k in STATS_FIELDS_RANGES:
                    # Take a 10% step in a random direction
                    vmin, vmax = STATS_FIELDS_RANGES[k]
                    step_size = (vmax - vmin) * 0.1
                    new_value = v + step_size * self.rng.uniform(-1, 1)
                    ue[k] = STATS_FIELDS_TYPES[k](np.clip(new_value, vmin, vmax))

        # Mark this data as fake / simulated
        data["is_fake"] = True

        return data

    def start_publisher(self):
        """Start the publisher and send UE stats at regular intervals."""
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            self.log.info(f"UE Stats Publisher started on port {self.port}")
            self.log.info("Press Ctrl+C to stop")
            self.running = True

            # TODO: maybe wait from some kind of "start" message from the client
            # so that the queue is not full by the time the client connected?

            while self.running:
                try:
                    # Generate current UE stats
                    ue_stats = self._generate_live_ue_stats()

                    # Publish the data under the specified topic
                    self.socket.send_multipart(
                        [
                            self.topic.encode("utf-8"),
                            json.dumps(ue_stats).encode("utf-8"),
                        ]
                    )

                    self.log.info(
                        f"Published UE stats for {len(ue_stats['UE_stats'])} UEs at {ue_stats['timestamp']}"
                    )

                    # Wait for next interval
                    time.sleep(self.interval)

                except zmq.error.ZMQError as e:
                    self.log.error(f"ZMQ Error: {e}")
                    break
                except Exception as e:
                    self.log.error(f"Error publishing data: {e}")
                    time.sleep(self.interval)

        except KeyboardInterrupt:
            self.log.info("Publisher shutdown requested")
        finally:
            self._cleanup()

    def stop_publisher(self):
        """Stop the publisher."""
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        self.log.info("Cleaning up publisher resources...")
        self.running = False
        self.socket.close()
        self.context.term()
        self.log.info("Publisher stopped")
