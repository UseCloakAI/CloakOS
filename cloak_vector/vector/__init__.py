"""Our own Vector client, built on wire-pod: discovery over its HTTP API, then gRPC straight to the robot."""
from .client import VectorClient, rgb565
from .discovery import BotInfo, discover

__all__ = ["BotInfo", "VectorClient", "discover", "rgb565"]
