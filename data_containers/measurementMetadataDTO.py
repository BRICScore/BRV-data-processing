from config import *
import datetime

class MeasurementMetadataDTO:
    def __init__(self):
        self.person_id: str
        self.timestamp: float
        self.duration_ms: int
        self.labels: str
        