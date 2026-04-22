import os
import sys
import pickle
import numpy as np
from xgboost import XGBClassifier

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            # 🔍 Debug
            unique_classes = np.unique(y_train)
            logging.info(f"Unique classes in y_train: {unique_classes}")

            # 🚨 Ensure binary classification
            if len(unique_classes) != 2:
                raise Exception(f"Expected binary classes [0,1], got {unique_classes}")

            # ✅ Compute scale_pos_weight manually
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)

            if pos_count == 0:
                raise Exception("No positive class (1) found in training data")

            scale_pos_weight = neg_count / pos_count
            logging.info(f"scale_pos_weight: {scale_pos_weight}")

            # ✅ Model
            model = XGBClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                learning_rate=self.model_trainer_config.learning_rate,
                max_depth=self.model_trainer_config.max_depth,
                random_state=self.model_trainer_config.random_state,
                eval_metric=self.model_trainer_config.eval_metric,
                scale_pos_weight=scale_pos_weight
            )

            model.fit(X_train, y_train)
            logging.info("Model training completed")

            return model

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            # Load transformed data
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)

            # Train model
            model = self.train_model(X_train, y_train)

            # Save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            logging.info(
                f"Model saved at {self.model_trainer_config.trained_model_file_path}"
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path
            )

        except Exception as e:
            raise MyException(e, sys)