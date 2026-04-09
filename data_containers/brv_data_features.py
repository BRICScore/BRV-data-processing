from dataclasses import dataclass, field
from numpy import ndarray, float64

@dataclass
class BRVDataFeatures:

    bpm:                            ndarray[tuple[int,], float64] = None
    avg_breath_depth:               ndarray[tuple[int,], float64] = None
    avg_breath_depth_std_dev:       ndarray[tuple[int,], float64] = None
    phases_avg_values:              ndarray[tuple[int, 4], float64] = None
    breath_shape:                   ndarray[tuple[int, 4], float64] = None
    breath_length_variability:      ndarray[tuple[int,], float64] = None
    breath_amplitude_variability:   ndarray[tuple[int,], float64] = None
    belt_share:                     ndarray[tuple[int, 5], float64] = None
    belt_share_std:                 ndarray[tuple[int, 5], float64] = None