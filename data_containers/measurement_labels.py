from dataclasses import dataclass, field
from brics_types import ActivityType
from .bio_data import BioData

@dataclass
class MeasurementLabels:
    activity:       ActivityType = None
    person_data:    BioData = field(default_factory=BioData)

