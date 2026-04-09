from dataclasses import dataclass, field
from .measurement_labels import MeasurementLabels
from pathlib import Path

@dataclass
class MeasurementMetadata:
    _id:                str = None
    timestamp:          float = None
    duration_ms:        int = None
    filepath_raw:       Path = None
    filepath_clean:     Path = None
    filepath_features:  Path = None
    labels:             MeasurementLabels = field(default_factory=MeasurementLabels)
