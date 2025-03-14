from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    # Update the nan entries with the correct weekday values
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        df = X.copy()
        # Convert 'dteday' to datetime before using .dt accessor
        df['dteday'] = pd.to_datetime(df['dteday'])

        wkday_null_idx = df[df[self.variables].isnull() == True].index # Returns all the index that have weekday value as null aka nan
        df.loc[wkday_null_idx, self.variables] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3]) # Get the day as name

        return df


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    # Assign the fill value as the mode of weathersit
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
         # YOUR CODE HERE
        self.fill_value=X[self.variables].mode()[0]
        return self

    # Replace all the nan values with mode value derived in fit method
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)
        # Convert to string
        X[self.variables] = X[self.variables].astype(str)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    # The mapping will contain the values to map
    #   - key is current categorical variable value
    #   - value is the integer value to replace with
    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Map the value and set the type as int
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    # Has to take a list of numerical fields and work on them
    def __init__(self, variables: list):
        # YOUR CODE HERE
        if not isinstance(variables, list): # Changed to list
            raise ValueError("variables should be a list")

        self.variables = variables

    # Have to maintain a list of bounds to cover all numerical features
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        # Calculate lower and upper bounds for all numerical features
        self.bounds = {}
        for var in self.variables:
            q1 = X[var].quantile(0.25)
            q3 = X[var].quantile(0.75)
            iqr = q3 - q1
            self.bounds[var] = {
                'lower_bound': q1 - (1.5 * iqr),
                'upper_bound': q3 + (1.5 * iqr)
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        df = X.copy()

        for var in self.variables:
            # Set upperbound if x is greater than upperbound
            df[var] = df[var].apply(lambda x: self.bounds[var]['upper_bound'] if x > self.bounds[var]['upper_bound'] else x)
            # Set lowerbound if x is less than lowerbound
            df[var] = df[var].apply(lambda x: self.bounds[var]['lower_bound'] if x < self.bounds[var]['lower_bound'] else x)
        return df



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[[self.variables]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        # Do the one shot encoding transformation
        encoded_weekday = self.encoder.transform(X[[self.variables]])

        # Get the encoded feature names
        enc_wkday_features = self.encoder.get_feature_names_out()

        # Append encoded weekday features to X
        X[enc_wkday_features] = encoded_weekday
        #encoded_weekday_df = pd.DataFrame(encoded_weekday, columns=self.encoder.get_feature_names_out())
        #X = pd.concat([X, encoded_weekday_df], axis=1)

        return X


# Class to drop unwanted columns
class DropColumns(BaseEstimator, TransformerMixin):
    """ Drop unwanted columns """
    def __init__(self, variables: list):
        # YOUR CODE HERE
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.drop(self.variables, axis=1)
        return X
