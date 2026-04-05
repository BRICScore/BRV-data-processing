from dataclasses import dataclass
from .measurement_metadata import MeasurementMetadata
from .brv_data_clean import BRVDataClean
from .brv_data_features import BRVDataFeatures

@dataclass
class MeasurementData:
    metadata: MeasurementMetadata
    data_clean: BRVDataClean
    data_features: BRVDataFeatures
