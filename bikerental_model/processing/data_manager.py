import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline

from bikerental_model import __version__ as _version
from bikerental_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

# 1. Working on dteday column to extract year and month
def get_year_and_month(dataframe):

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    return df


# 2. processing cabin
f1 = lambda x: 1 if type(x) == str else 0         ## Ternary Expression


def numerical_feat():
    data_frame = load_dataset(file_name=config.app_config_.training_data_file)
    numerical_features = []
    categorical_features = []
    for col in data_frame.columns:
        if col not in [config.model_config_.target] + config.model_config_.unused_fields:
            if data_frame[col].dtypes == 'float64':
                numerical_features.append(col)
            else:
                categorical_features.append(col)
    
    return numerical_features, categorical_features

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame = get_year_and_month(data_frame)       # addition of year and month.

    return data_frame


def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

