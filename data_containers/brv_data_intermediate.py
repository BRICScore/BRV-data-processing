from pydantic import BaseModel, Field, ConfigDict
from numpy import float64, int64, array
from numpy.typing import NDArray
class BRVDataIntermediate(BaseModel): 
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    
    timestamps:             NDArray[int64] = Field(default_factory=lambda: array([], dtype=int64))
    adc_normalized_data:    NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    signal_minima:          NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))
    signal_maxima:          NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))