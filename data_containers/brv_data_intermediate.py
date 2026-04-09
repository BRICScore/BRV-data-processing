from dataclasses import dataclass
from numpy import float64, int64, ndarray

@dataclass
class BRVDataIntermediate: 
    
    timestamps:             ndarray[tuple[int,], int64] = None
    adc_normalized_data:    ndarray[tuple[int,5],float64] = None
    signal_minima:          ndarray[tuple[int,], int64] = None
    signal_maxima:          ndarray[tuple[int,], int64] = None