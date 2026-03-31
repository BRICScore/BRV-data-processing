from dataclasses import dataclass
from numpy import ndarray, float64

@dataclass
class BRVDataIntermediate: 
    timestamps: ndarray[float]
    adc_data: ndarray[tuple[int, 5], float64]
    signal_minima: ndarray[int]
    signal_maxima: ndarray[int]