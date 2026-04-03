from dataclasses import dataclass
from typing import Any
from numpy import float64, int64
from numpy.typing import NDArray

@dataclass
class BRVDataIntermediate: 
    def __init__(self):
        pass
    
    # TODO: define ndarray of lists of adc_data
    timestamps: NDArray[float64] 
    adc_normalized_data: list[NDArray[Any]]
    signal_minima: NDArray[int64]
    signal_maxima: NDArray[int64]