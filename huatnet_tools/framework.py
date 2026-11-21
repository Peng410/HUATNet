"""Compose temporal and contextual encoders for RUL prediction."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RULPrediction:
    """Point estimate and predictive standard deviation in cycles."""

    estimate: Any
    standard_deviation: Any


@dataclass(frozen=True)
class HUATNetScaffold:
    """Connect two encoding branches to a probabilistic prediction stage."""

    encode_temporal: Callable[[Any, Any], Any]
    encode_context: Callable[[Any, Any, Any], Any]
    predict_from_features: Callable[[Any, Any, Any], RULPrediction]

    def __call__(self, sensors, operating_settings, valid_mask) -> RULPrediction:
        """Accept aligned sensor windows, operating settings, and an observation mask."""
        temporal = self.encode_temporal(sensors, valid_mask)
        context = self.encode_context(sensors, operating_settings, valid_mask)
        return self.predict_from_features(temporal, context, valid_mask)
