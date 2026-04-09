from dataclasses import dataclass
from numpy import float64, ndarray, int64

@dataclass
class BRVDataClean:
    timestamps:     ndarray[tuple[int,], int64] = None
    adc_data:       ndarray[tuple[int, 5], float64] = None