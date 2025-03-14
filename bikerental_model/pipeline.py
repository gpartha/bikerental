import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer
from bikerental_model.processing.features import Mapper
from bikerental_model.processing.features import WeathersitImputer
from bikerental_model.processing.features import OutlierHandler
from bikerental_model.processing.features import WeekdayOneHotEncoder
from bikerental_model.processing.features import DropColumns
from bikerental_model.processing.data_manager import numerical_feat

numerical_features, categorical_features = numerical_feat()

# dteday,season,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,
# casual,registered,cnt,yr, mnth

bikeshare_pipeline = Pipeline([
    ('weekday_imputer', WeekdayImputer(config.model_config_.Weekday)),
    ('weather_situation', WeathersitImputer("weathersit")),
    # mapper
    ('map_yr', Mapper('yr', config.model_config_.yr_mapping)),
    ('map_month', Mapper('mnth', config.model_config_.mnth_mapping)),
    ('map_season', Mapper('season', config.model_config_.season_mapping)),
    ('map_weather', Mapper('weathersit', config.model_config_.weather_mapping)),
    ('map_holiday', Mapper('holiday', config.model_config_.holiday_mapping)),
    ('map_workingday', Mapper('workingday', config.model_config_.workingday_mapping)),
    ('map_hour', Mapper('hr', config.model_config_.hour_mapping)),
    # Outlier handler
    ('outlier_handler', OutlierHandler(numerical_features)),
    # One-hot encoder
    ('weekday_encoder', WeekdayOneHotEncoder("weekday")),
    ('drop_columns', DropColumns(config.model_config_.unused_fields)),
    # Regressor
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])
