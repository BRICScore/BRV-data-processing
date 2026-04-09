from dataclasses import dataclass, field
from numpy import float64, int64, array
from numpy.typing import NDArray

@dataclass
class BRVDataIntermediate: 
    
    timestamps:             NDArray[int64] = field(default_factory=lambda: array([], dtype=int64))
    adc_normalized_data:    NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    signal_minima:          NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    signal_maxima:          NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))