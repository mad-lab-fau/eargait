"""Eargait: The Gait and Movement Analysis Package unsing Ear-Worn IMU sensors."""
from eargait.event_detection.base_event_detection import BaseEventDetection
from eargait.event_detection.diao_adapted_event_detection import DiaoAdaptedEventDetection
from eargait.event_detection.diao_event_detection import DiaoEventDetection

__all__ = ["BaseEventDetection", "DiaoEventDetection", "DiaoAdaptedEventDetection"]
