from pydantic import BaseModel, Field, ConfigDict
from numpy import array, float64, int64
from numpy.typing import NDArray
from typing import Optional

class BRVDataClean(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    timestamps:     NDArray[int64] = Field(default_factory=lambda: array([], dtype=int64))
    adc_data:       NDArray[float64] = Field(default_factory=lambda: array([], dtype=float64))