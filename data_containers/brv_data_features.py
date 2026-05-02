from pydantic import BaseModel, Field, ConfigDict
from numpy import float64, array
from numpy.typing import NDArray

class BRVDataFeatures(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    bpm:                            NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    avg_breath_depth:               NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    avg_breath_depth_std_dev:       NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    phases_avg_values:              NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    breath_shape:                   NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    breath_length_variability:      NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    breath_amplitude_variability:   NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    belt_share:                     NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    belt_share_std:                 NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))