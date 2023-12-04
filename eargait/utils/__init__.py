"""Eargait: The Gait and Movement Analysis Package unsing Ear-Worn IMU sensors."""
from eargait.utils.gravity_alignment import StaticWindowGravityAlignment, TrimMeanGravityAlignment

__all__ = ["TrimMeanGravityAlignment", "StaticWindowGravityAlignment"]
