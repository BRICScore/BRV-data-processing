from dataclasses import dataclass

from brics_types import ConditionType, GenderType

@dataclass
class BioData:
    age : int
    gender : GenderType
    health: str
    condition: ConditionType
    weight: int
    height: int