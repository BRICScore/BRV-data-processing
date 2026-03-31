from numpy import ndarray, float64
from dataclasses import dataclass

@dataclass
class BRVDataClean:
    timestamps: ndarray[tuple[int,], float64]
    adc_data: ndarray[tuple[int, 5], float64]
