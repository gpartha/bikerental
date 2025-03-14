import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikerental_model import __version__ as _version
from bikerental_model.config.core import config
from bikerental_model.processing.data_manager import load_pipeline
from bikerental_model.processing.data_manager import pre_pipeline_preparation
from bikerental_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bikerental_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data, index=[0]))
    
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bikerental_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bikerental_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":
    data_in = {"dteday":"2012-11-05",
               "season": "spring", 
               "hr": "4am", 
               "holiday":'No',
               "weekday":'Mon', 
               "workingday":'Yes',
               "weathersit":'Clear',
               "temp": -10,
               "atemp": -12.1, 
               "hum": 49,
               "windspeed": 19,
               "casual": 4,
               "registered": 135
               }
    
    res = make_prediction(input_data=data_in)
    print(res)
