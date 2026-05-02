from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from brics_types import ConditionType, GenderType

class BioData(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    person_id:  str = ""
    age :       int = 0
    gender :    Optional[GenderType] = None
    health:     str = ""
    condition:  Optional[ConditionType] = None
    weight:     int = 0
    height:     int = 0