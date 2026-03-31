from dataclasses import dataclass
from pathlib import Path
from .MeasurementMetadata import MeasurementMetadata
from .BRVDataClean import BRVDataClean
from .BRVDataFeatures import BRVDataFeatures

@dataclass
class MeasurementData:
    metadata: MeasurementMetadata
    data_clean: BRVDataClean
    data_features: BRVDataFeatures
    filepath_raw: Path
    filepath_clean: Path
    filepath_features: Path

