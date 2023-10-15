# import polars as pl
import pandas as pd


def load_data_frame(data_path: str) -> pd.DataFrame:
    """Loading data with some necessary preprocessing"""

    return pd.read_csv(data_path)


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
