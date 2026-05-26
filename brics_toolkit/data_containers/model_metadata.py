from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

class ModelMetadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    id:                 str = Field(default="", alias="_id")
    timestamp:          float = 0.0
    filepath_weights:   Path = Path()
    filepath_pth:       Path = Path()
    filepath_scaler:    Path = Path()
