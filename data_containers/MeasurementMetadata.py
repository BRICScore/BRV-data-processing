from dataclasses import dataclass
from .MeasurementLabels import MeasurementLabels
from pathlib import Path

@dataclass
class MeasurementMetadata:
    _id: str
    timestamp: float
    duration_ms: int
    filepath_raw: Path
    filepath_clean: Path
    filepath_features: Path
    labels: MeasurementLabels
