from dataclasses import dataclass
from brics_types import ActivityType
from .BioData import BioData

@dataclass
class MeasurementLabels:
    activity: ActivityType
    person_data: BioData

