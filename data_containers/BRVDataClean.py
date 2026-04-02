from dataclasses import dataclass
from typing import Any
from numpy import float64, int64
from numpy.typing import NDArray


@dataclass
class BRVDataClean:
    def __init__(self):
        pass

    timestamps: NDArray[float64]
    adc_data: list[NDArray[Any]]
