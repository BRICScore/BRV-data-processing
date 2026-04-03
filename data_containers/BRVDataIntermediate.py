from dataclasses import dataclass
from numpy import float64, int64
from numpy.typing import NDArray

@dataclass
class BRVDataIntermediate: 
    def __init__(self):
        pass
    
    timestamps: NDArray[float64] 
    adc_normalized_data: NDArray[float64]
    signal_minima: NDArray[int64]
    signal_maxima: NDArray[int64]