from __future__ import annotations

from typing import Dict, Any

import zmq

from .srk_client_base import SrkClientBase, ReceiveResult


STATS_FIELDS_NAMES = {
    "timestamp": "Timestamp",
    "rnti": "RNTI",
    "mcs": "MCS",
    "bler": "BLER",
    "num_prbs": "PRBs",
    "mcs_up": "MCS up",
    "mcs_down": "MCS down",
    "bler_up": "BLER up",
    "bler_down": "BLER down",
    "num_prbs_up": "PRBs up",
    "num_prbs_down": "PRBs down",
    "goodput": "Goodput (Mbit/s)",
    "goodput_up": "Goodput up (Mbit/s)",
    "goodput_down": "Goodput down (Mbit/s)",
}
STATS_FIELDS_TYPES = {
    "timestamp": int,
    "rnti": int,
    "mcs_up": int,
    "mcs_down": int,
    "bler_up": float,
    "bler_down": float,
    "num_prbs_up": int,
    "num_prbs_down": int,
    "goodput_up": float,
    "goodput_down": float,
}
STATS_FIELDS_RANGES = {
    "rnti": (0, 65535),
    "mcs_up": (0, 28),
    "mcs_down": (0, 28),
    "bler_up": (0, 1),
    "bler_down": (0, 1),
    "num_prbs_up": (0, 51),
    "num_prbs_down": (0, 51),
    "goodput_up": (0, 30),
    "goodput_down": (0, 30),
}

}


class UEStatsSubscriber(SrkClientBase):
    """
    Helper class to receive UE statistics from a ZeroMQ server.
    """

    def __init__(self):
        """
        Initialize the ZeroMQ UE stats subscriber.
        """
        super().__init__("UE stats", zmq.SUB)

    def try_receive(self) -> tuple[ReceiveResult, dict | Exception | None]:
        status, result = super().try_receive()
        if status == ReceiveResult.SUCCESS:
            # Enforce dtypes
            for k, v in result.items():
                if k in STATS_FIELDS_TYPES:
                    result[k] = STATS_FIELDS_TYPES[k](v)

            stats = result.get("UE_stats", [])
            for ue in stats:
                for kk, vv in ue.items():
                    if kk in STATS_FIELDS_TYPES:
                        ue[kk] = STATS_FIELDS_TYPES[kk](vv)

        return status, result

    def _process_ue_stats(self, ue_stats: Dict[str, Any]):
        """Process received UE statistics."""
        try:
            self.message_count += 1
            timestamp = ue_stats.get("timestamp", 0)
            num_ues = len(ue_stats.get("UE_stats", []))

            self.log.debug(
                f"Received UE stats #{self.message_count}: {num_ues} UEs at timestamp {timestamp}"
            )
            self._print_ue_details(ue_stats)

        except Exception as e:
            self.log.error(f"Error processing UE stats: {e}")

    def _print_ue_details(self, ue_stats: Dict[str, Any]):
        """Print UE details."""
        print(
            f"\n--- UE Stats #{self.message_count} (TS: {ue_stats.get('timestamp', 'N/A')}) ---"
        )

        for ue in ue_stats.get("UE_stats", []):
            print(
                f"{ue.get('ue_id')}, RNTI {ue.get('rnti')}: MCS {ue.get('mcs_up')}/{ue.get('mcs_down')}, "
                f"BLER {ue.get('bler_up', 0):.3f}/{ue.get('bler_down', 0):.3f}, "
                f"PRBs {ue.get('num_prbs_up')}/{ue.get('num_prbs_down')}"
            )

        print("---")
