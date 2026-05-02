from pydantic import BaseModel, Field, ConfigDict
from .measurement_metadata import MeasurementMetadata
from .brv_data_clean import BRVDataClean
from .brv_data_features import BRVDataFeatures

class MeasurementData(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    metadata:       MeasurementMetadata = Field(default_factory=MeasurementMetadata)
    data_clean:     BRVDataClean = Field(default_factory=BRVDataClean)
    data_features:  BRVDataFeatures = Field(default_factory=BRVDataFeatures)