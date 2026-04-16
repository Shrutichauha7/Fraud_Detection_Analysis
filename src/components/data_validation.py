import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_columns(self, dataframe: DataFrame) -> bool:
        try:
            schema_columns = [list(col.keys())[0] for col in self._schema_config["columns"]]
            dataframe_columns = list(dataframe.columns)

            missing_columns = [col for col in schema_columns if col not in dataframe_columns]
            extra_columns = [col for col in dataframe_columns if col not in schema_columns]

            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
            if extra_columns:

                if "_id" in extra_columns:
                        extra_columns.remove("_id")

            return len(missing_columns) == 0

        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns

            missing_numerical_columns = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]

            missing_categorical_columns = [
                col for col in self._schema_config["categorical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = []
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            drop_cols = self._schema_config.get("drop_columns", [])
            if drop_cols:
                train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
                test_df.drop(columns=drop_cols, inplace=True, errors="ignore")
                logging.info(f"Dropped columns: {drop_cols}")

            # Validate columns
            if not self.validate_columns(train_df):
                validation_error_msg.append("Missing columns in training dataframe.")
            else:
                logging.info("Training dataframe columns validated.")

            if not self.validate_columns(test_df):
                validation_error_msg.append("Missing columns in test dataframe.")
            else:
                logging.info("Testing dataframe columns validated.")

            # Validate column existence
            if not self.is_column_exist(train_df):
                validation_error_msg.append("Columns missing in training dataframe.")

            if not self.is_column_exist(test_df):
                validation_error_msg.append("Columns missing in test dataframe.")

            # Final status
            validation_status = len(validation_error_msg) == 0

            final_message = (
                "Validation successfully completed"
                if validation_status
                else " ".join(validation_error_msg)
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=final_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump({
                    "validation_status": validation_status,
                    "message": final_message
                }, report_file, indent=4)

            logging.info("Data validation completed.")
            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys)