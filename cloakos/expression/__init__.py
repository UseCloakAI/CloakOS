"""Cloak's own expression layer: procedural eyes, synthesized chirps, motion emotes."""
from .chirps import chirp
from .emotes import Emotes
from .eyes import EyeShape, Eyes

__all__ = ["Emotes", "EyeShape", "Eyes", "chirp"]
