from dataclasses import dataclass, field
from numpy import float64, array
from numpy.typing import NDArray

@dataclass
class BRVDataFeatures:

    bpm:                            NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    avg_breath_depth:               NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    avg_breath_depth_std_dev:       NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    phases_avg_values:              NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    breath_shape:                   NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    breath_length_variability:      NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    breath_amplitude_variability:   NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    belt_share:                     NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))
    belt_share_std:                 NDArray[float64] = field(default_factory=lambda: array([], dtype=float64))