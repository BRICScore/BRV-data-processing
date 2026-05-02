from pydantic import BaseModel, Field, ConfigDict
from brics_types import ActivityType
from .bio_data import BioData
from typing import Optional

class MeasurementLabels(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    activity:       Optional[ActivityType] = None
    person_data:    BioData = Field(default_factory=BioData)

