from __future__ import annotations

from enum import Enum
import json
import logging
from typing import Dict, Any
import socket
import weakref

import zmq


class ReceiveResult(Enum):
    SUCCESS = 0
    NO_MESSAGE = 1
    DECODE_ERROR = 2
    OTHER_ERROR = 3


class SrkClientBase:
    """
    Base class for ZeroMQ-based clients used in the SRK demo.
    """

    def __init__(self, name: str, socket_type: zmq.SocketType):
        """
        Initialize the ZeroMQ client.

        Args:
            sleep_delay_s: How long to sleep between queue polls, in seconds.
        """
        self.log = logging.getLogger(__name__)
        self.name = name

        self.client_id = socket.gethostname()

        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket_type = socket_type
        self.socket = None
        self.topic: str | None = None

        # Control variables
        self.message_count = 0

        # Make sure the resources are cleaned up when the object is garbage collected.
        weakref.finalize(self, self._cleanup)

    def connect(
        self, server_host: str, server_port: int, topic: str | None = None
    ) -> bool:
        """
        Connect to the server.

        Args:
            server_host: Server hostname or IP address
            server_port: Server port number
        Returns:
            True if connection successful, False otherwise
        """
        try:
            server_url = f"tcp://{server_host}:{server_port}"
            self.log.debug(f"Connecting to {self.name} server at {server_url}...")
            self.socket = self.context.socket(self.socket_type)

            # Subscribe to topic, if given
            if topic is not None:
                self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
                self.topic = topic

            # Receive timeout: 5 seconds
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            # Do not wait before terminating the context
            self.socket.setsockopt(zmq.LINGER, 0)

            self.socket.connect(server_url)

            self.log.info(
                f"Listening to {self.name} server at {server_url} (topic: {topic})"
            )
            return True

        except Exception as e:
            self.log.error(f"Failed to connect to publisher: {e}")
            self.disconnect()
            return False

    def reconnect(self):
        """Reconnect to the server."""
        # Get current host, port and topic
        host = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8").strip("tcp://")
        host, port = host.split(":")
        self.disconnect()
        self.connect(host, port, self.topic)

    def disconnect(self):
        """Disconnect from the UE stats publisher."""
        if self.socket is not None:
            self.socket.close()
        self.socket = None
        self.topic = None
        self.log.info(f"Disconnected from {self.name} server")

    def is_connected(self) -> bool:
        """Check if the client is connected to the UE stats publisher."""
        return self.socket is not None

    def try_receive(self) -> tuple[ReceiveResult, dict | Exception | None]:
        try:
            if self.topic is not None:
                # Receive multipart message
                message_parts = self.socket.recv_multipart(zmq.NOBLOCK)

                if len(message_parts) >= 2:
                    loaded = json.loads(message_parts[1].decode("utf-8"))
                else:
                    return ReceiveResult.DECODE_ERROR, message_parts
            else:
                loaded = self.socket.recv_json(zmq.NOBLOCK)

            self.message_count += 1
            return ReceiveResult.SUCCESS, loaded

        except zmq.error.Again:
            return ReceiveResult.NO_MESSAGE, None
        except json.JSONDecodeError as e:
            return ReceiveResult.DECODE_ERROR, e
        except Exception as e:
            return ReceiveResult.OTHER_ERROR, e

    def get_statistics(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        return {
            "messages_received": self.message_count,
            "client_id": self.client_id,
        }

    def _cleanup(self):
        """Clean up resources."""

        self.log.info(f"Cleaning up {self.name} client resources...")
        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        self.log.info(
            f"{self.name} client stopped. Total messages received: {self.message_count}"
        )
