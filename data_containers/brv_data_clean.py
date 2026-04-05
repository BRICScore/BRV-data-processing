from dataclasses import dataclass
from numpy import float64
from numpy.typing import NDArray

@dataclass
class BRVDataClean:
    def __init__(self):
        pass

    timestamps: NDArray[float64]
    adc_data: NDArray[float64]