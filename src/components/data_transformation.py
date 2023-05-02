import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# The @dataclass decorator is used to create a simple class that primarily holds data, similar to a struct in C
# When you apply @dataclass to a class, the decorator automatically generates some special methods such as __init__, __repr__,
# By using the @dataclass decorator, the class is automatically provided with an implementation of the __init__ method 
# that takes a single argument for each attribute defined in the class, along with other special methods.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        """
        Function that handles data transformation
        """
        try:
            logging.info("Strating data transformation")
            numeric_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]    
            )
            logging.info("Data transformation compleated for numerical columns")
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]  
            )
            logging.info("Data transformation compleated for categorical columns")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, cat_features),
                ]
            )
            logging.info("Data transformation compleated")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # read train and test data as Pandas dataframes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # log the event of loading train and test data
            logging.info("Train and thest data loaded")

            # get the preprocessor object that performs data transformation
            preprocessing_obj = self.get_data_transform_object()

            # specify the name of the target column in the dataset
            target_column_name ="math_score"

            # extract input features and target variable from train data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            # extract input features and target variable from test data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # perform data transformation on input features of train and test data
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # combine input features and target variable into numpy arrays for train and test data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # save the preprocessor object to a file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            # return the transformed train and test data, and the file path to the preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        # catch any exceptions that occur during the data transformation process
        except Exception as e:
            raise CustomException(e, sys)
