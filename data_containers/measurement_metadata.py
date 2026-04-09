from dataclasses import dataclass, field
from .measurement_labels import MeasurementLabels
from pathlib import Path
from typing import Optional

@dataclass
class MeasurementMetadata:
    _id:                str = ""
    timestamp:          float = 0.0
    duration_ms:        int = 0
    filepath_raw:       Path = Path()
    filepath_clean:     Path = Path()
    filepath_features:  Path = Path()
    labels:             MeasurementLabels = field(default_factory=MeasurementLabels)
