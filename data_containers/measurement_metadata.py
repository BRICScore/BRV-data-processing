from pydantic import BaseModel, Field, ConfigDict
from .measurement_labels import MeasurementLabels
from pathlib import Path

class MeasurementMetadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    id:                 str = Field(default="", alias="_id")
    timestamp:          float = 0.0
    duration_ms:        int = 0
    filepath_raw:       Path = Path()
    filepath_clean:     Path = Path()
    filepath_features:  Path = Path()
    labels:             MeasurementLabels = Field(default_factory=MeasurementLabels)
