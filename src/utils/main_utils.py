import os
import sys
import numpy as np
import dill
import yaml
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging



def read_yaml_file(file_path: str) -> dict:
    try:
        logging.info(f"Reading YAML file from: {file_path}")
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        logging.info(f"Writing YAML file to: {file_path}")
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise MyException(e, sys) from e



def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        logging.info(f"Loading object from: {file_path}")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise MyException(e, sys) from e



def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        logging.info(f"Saving numpy array to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise MyException(e, sys) from e



def load_numpy_array_data(file_path: str) -> np.array:
    try:
        logging.info(f"Loading numpy array from: {file_path}")

        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise MyException(e, sys) from e



def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop columns from a pandas DataFrame
    """
    try:
        logging.info(f"Dropping columns: {cols}")

        return df.drop(columns=cols, axis=1, errors="ignore")

    except Exception as e:
        raise MyException(e, sys) from e



def separate_features_target(df: DataFrame, target_column: str):
    try:
        logging.info(f"Separating target column: {target_column}")

        X = df.drop(columns=[target_column], axis=1)
        y = df[target_column]

        return X, y

    except Exception as e:
        raise MyException(e, sys) from e



def save_dataframe(file_path: str, df: DataFrame):
    try:
        logging.info(f"Saving dataframe to: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

    except Exception as e:
        raise MyException(e, sys) from e



def load_dataframe(file_path: str) -> DataFrame:
    try:
        logging.info(f"Loading dataframe from: {file_path}")

        return pd.read_csv(file_path)

    except Exception as e:
        raise MyException(e, sys) from e
    
def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise MyException(e, sys) from e