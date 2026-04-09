from dataclasses import dataclass, field
from numpy import array, float64, int64
from numpy.typing import NDArray
from typing import Optional

@dataclass
class BRVDataClean:
    timestamps:     NDArray[int64] = field(default_factory=lambda: array([], dtype=int64))
    adc_data:       NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))