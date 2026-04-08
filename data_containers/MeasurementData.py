from dataclasses import dataclass
from .MeasurementMetadata import MeasurementMetadata
from .BRVDataClean import BRVDataClean
from .BRVDataFeatures import BRVDataFeatures

class MeasurementData:
    metadata: MeasurementMetadata
    data_clean: BRVDataClean
    data_features: BRVDataFeatures
