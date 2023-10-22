# built-in imports
# 3p imports
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# project imports


def load_data_frame(data_path: str, n_sample: int = None, drop_na_ratio: float = None) -> tuple[pd.DataFrame, pd.DataFrame]:

    df: pd.DataFrame = pd.read_csv(data_path)

    if n_sample is not None:
        df = df.sample(n=n_sample)
        print(f"From the original data, {n_sample} samples was created!")
    else:
        n_sample = len(df)

    # initialize target vector and feature matrix
    y = df["TARGET"]
    X = df.drop(["TARGET"], axis=1)

    if drop_na_ratio is not None:

        null_count_df: pd.Series = df.isna().sum()
        drop_null_count_df: pd.Series = null_count_df[null_count_df/n_sample >= drop_na_ratio]
        X.drop(drop_null_count_df.index, axis=1)
        print(f"From the original data, {len(drop_null_count_df)} features was dropped!")


    return X, y


def flag_columns_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags are loaded as a simple string. This function replaces them with
    {"Y": True, "N": False} mapping."""

    str_mapping = {"Y": True, "N": False}
    int_mapping = {1: True, 0: False}

    all_column_name_list = df.columns.values.tolist()
    flag_column_name_list = [
        column_name for column_name in all_column_name_list if "FLAG" in column_name
    ]

    for column_name in flag_column_name_list:
        column_data = df[column_name]
        data_type = column_data.dtype

        if str(data_type) != "bool":
            match str(data_type):
                case "object":
                    new_column_data = [
                        str_mapping[value] for _, value in column_data.items()
                    ]

                case "int64":
                    new_column_data = [
                        int_mapping[value] for _, value in column_data.items()
                    ]

            df[column_name] = new_column_data

    return df


def create_column_transformer(X: pd.DataFrame) -> ColumnTransformer:
    """This function prepares the necessary column transformer to preprocess the data frame.

    Explanation:
        - for all the data, there may be some missing datums, which are represented by np.NaN, these 
        are presented to simple imputer, that replaces them by
            - meadian of all the valid feature data, since we need the resulting value to be one of  
            the others and thus an integer value
            - mean of all the valid feature data (there may be a huge statistical dispute on why the 
            mean is the least appropriate value) for float-like values
            - most frequent value of all the valid feature data for categorical values

    Args:
        X (pd.DataFrame): _description_

    Returns:
        ColumnTransformer: _description_
    """
    return make_column_transformer(
        (
            SimpleImputer(missing_values=np.NaN, strategy="median"),
            list(X.select_dtypes(include="int64").columns),
        ),
        (
            make_pipeline(
                SimpleImputer(missing_values=np.NaN, strategy="mean"), StandardScaler()
            ),
            list(X.select_dtypes(include="float64").columns),
        ),
        (
            make_pipeline(
                OneHotEncoder(handle_unknown="ignore"),
                SimpleImputer(missing_values=np.NaN, strategy="most_frequent"),
            ),
            list(X.select_dtypes(include="object")),
        ),
    )
