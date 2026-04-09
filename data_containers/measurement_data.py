from dataclasses import dataclass, field
from .measurement_metadata import MeasurementMetadata
from .brv_data_clean import BRVDataClean
from .brv_data_features import BRVDataFeatures

@dataclass
class MeasurementData:

    metadata:       MeasurementMetadata = field(default_factory=MeasurementMetadata)
    data_clean:     BRVDataClean = field(default_factory=BRVDataClean)
    data_features:  BRVDataFeatures = field(default_factory=BRVDataFeatures)