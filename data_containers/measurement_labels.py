from dataclasses import dataclass
from brics_types import ActivityType
from .bio_data import BioData

@dataclass
class MeasurementLabels:
    activity: ActivityType
    person_data: BioData

