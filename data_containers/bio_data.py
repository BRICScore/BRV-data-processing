from dataclasses import dataclass
from typing import Optional

from brics_types import ConditionType, GenderType

@dataclass
class BioData:
    person_id:  str = ""
    age :       int = 0
    gender :    Optional[GenderType] = None
    health:     str = ""
    condition:  Optional[ConditionType] = None
    weight:     int = 0
    height:     int = 0