from dataclasses import dataclass

from brics_types import ConditionType, GenderType

@dataclass
class BioData:
    person_id:  str = None
    age :       int = None
    gender :    GenderType = None
    health:     str = None
    condition:  ConditionType = None
    weight:     int = None
    height:     int = None