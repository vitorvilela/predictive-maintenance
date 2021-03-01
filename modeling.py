import sys
import os
import numpy as np
import csv

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from joblib import dump, load

import analysis



class PredictionPipeline:

        cwd = os.getcwd()

        def __init__(self, log, transformed_dataset, model, chosen_model_name, output_filename):
                """
                Info
                """

                self.log = log
                self.args = log.args

                self.transformed_dataset = transformed_dataset

                self.scaler = model.data_preprocess_pipeline.scaler                
                self.features_name = model.data_preprocess_pipeline.features_name
                self.models = model.models
                
                self.chosen_model_name = chosen_model_name
                self.output_filename = output_filename

                self.set_prediction()
                self.output_csv()


        def set_prediction(self):
                """
                Info
                """

                df = self.transformed_dataset.dataframe

                if not self.args.option_test_dataset_cutoff:
                        self.data_id_array = df['data-id'].values
                        self.x = df[self.features_name].values                
                else:
                        self.data_id_array = df.loc[df['monitoring-cycle']>100]['data-id'].values
                        self.x = df.loc[df['monitoring-cycle']>100][self.features_name].values
                                
                x_preprocessed = self.scaler.transform(self.x)
                                                                         
                self.y_pred = self.models[self.chosen_model_name].predict(x_preprocessed).astype(int)
                
                pred_analysis = analysis.Analysis(log=self.log)
                pred_analysis.log_violinchart(self.y_pred, 'rul_prediction', self.output_filename.split('.csv')[0])
                

        def output_csv(self):
                """
                Info
                """

                cls = self.__class__

                
                file_path = os.path.join(os.path.join(cls.cwd, 'output'), self.output_filename)

                with open(file_path, mode='w') as output_file:
                        csv_writer = csv.writer(output_file, delimiter=',')
                        for d, p in zip(self.data_id_array, self.y_pred):
                                csv_writer.writerow([d, p])                               






                








class DataPreprocessPipeline:

        def __init__(self, log, transformed_dataset):
                """
                Info
                """

                self.log = log
                self.args = log.args
                self.transformed_dataset = transformed_dataset
                self.degree = self.args.polynomial_features_degree

                #
                #if self.transformed_dataset.type=='train':
                self.targets_name = ['rul']
                #else:
                #        self.targets_name = None      

                if self.args.option_use_derivatives:
                        self.numeric_features_name = ['monitoring-cycle'] + self.transformed_dataset.selected_settings + self.transformed_dataset.selected_sensors + self.transformed_dataset.selected_sensors_time_derivative
                else: 
                        self.numeric_features_name = ['monitoring-cycle'] + self.transformed_dataset.selected_settings + self.transformed_dataset.selected_sensors
                
                self.features_name = self.numeric_features_name
                                
                self.set_targets_features_array()
                self.set_polynomial_features()
                self.set_pipeline()
                self.set_scaler()
                


        def set_targets_features_array(self):
                """                
                Returns: a tuple
                """

                #if self.transformed_dataset.type == 'train':

                self.train_val_split_seed = 1
                self.val_ratio = 0.3

                x = self.transformed_dataset.dataframe[self.features_name].values
                y = self.transformed_dataset.dataframe[self.targets_name].values     

                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=self.val_ratio, random_state=self.train_val_split_seed) 

                #elif self.dataset.type=='test':

                #        x_test = self.transformed_dataset.dataframe[self.numeric_features_name]

                #else:
                #        print(f'\nIn create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.\n')
                #        sys.exit(1)               


        def set_polynomial_features(self): 
                """
                Info
                """

                self.polynomial_features = PolynomialFeatures(degree=self.degree, include_bias=False)


        def set_pipeline(self): 
                """
                Info
                """   

                if self.args.option_polynomial_features and self.args.option_standardize:   
                        self.numeric_pipeline = Pipeline(steps=[ ('polynomial_features', self.polynomial_features), ('Scaler', StandardScaler()) ])
                elif self.args.option_polynomial_features and self.args.option_standardize==False: 
                        self.numeric_pipeline = Pipeline(steps=[ ('polynomial_features', self.polynomial_features) ])
                elif self.args.option_polynomial_features==False and self.args.option_standardize: 
                        self.numeric_pipeline = Pipeline(steps=[ ('Scaler', StandardScaler()) ])  
                else:
                        print('\nOn get_pipeline(), it is not possible to create a data preprocess pipeline without at least option_polynomial_features or option_standardize set True.')
                        sys.exit(1)

                self.pipeline = self.numeric_pipeline


        def set_scaler(self):
                """
                Info
                """ 

                self.scaler = self.pipeline.fit(self.x_train)



class ModelingPipeline:

        def __init__(self, log, data_preprocess_pipeline):
                """
                Info
                """

                self.log = log
                self.args = log.args
                self.data_preprocess_pipeline = data_preprocess_pipeline

                self.set_pipeline()

                
        def set_pipeline(self):
                """
                Info
                """

                self.selected_models = [('DUMMY', DummyRegressor()), ('AB', AdaBoostRegressor()), ('GB', GradientBoostingRegressor()), ('RF', RandomForestRegressor()), ('ET', ExtraTreesRegressor())]

                self.pipeline = []

                for model_name, model in self.selected_models:
                        self.pipeline.append((model_name, Pipeline([('data_pipeline', self.data_preprocess_pipeline.pipeline), (model_name, model)])))
                      





class Model:

        def __init__(self, log, modeling_pipeline):
                """
                Info
                """

                self.log = log
                self.args = log.args
                self.modeling_pipeline = modeling_pipeline
                self.data_preprocess_pipeline = modeling_pipeline.data_preprocess_pipeline

                self.set_models()


        def set_models(self):
                """
                Info
                """

                self.models = {}

                x_train = self.data_preprocess_pipeline.x_train
                x_val = self.data_preprocess_pipeline.x_val
                y_train = self.data_preprocess_pipeline.y_train
                y_val = self.data_preprocess_pipeline.y_val

                for degree in range(1, self.args.polynomial_features_degree+1):

                        print(f'\nDegree: {degree}')

                        #self.data_preprocess_pipeline.set_polynomial_features(degree)
                        #self.data_preprocess_pipeline.set_pipeline()
                        #self.modeling_pipeline.set_pipeline()

                        for model_name, pipeline in self.modeling_pipeline.pipeline:   

                                self.models[model_name]= pipeline.fit(x_train, y_train)
                                
                                y_train_pred = pipeline.predict(x_train)
                                y_val_pred = pipeline.predict(x_val)
                                
                                # Log on stdout
                                print(f'\nModel: {model_name}')
                                
                                print(f'\nape (mean - max) | train / val')
                                ape = np.abs(100*(np.squeeze(y_train)-y_train_pred)/np.squeeze(y_train))
                                print(f'{np.mean(ape)} - {np.max(ape)}')
                                ape = np.abs(100*(np.squeeze(y_val)-y_val_pred)/np.squeeze(y_val))
                                print(f'{np.mean(ape)} - {np.max(ape)}')
                                
                                print(f'\nae (mean - max) | train / val')
                                ae = np.abs(np.squeeze(y_train)-y_train_pred)
                                print(f'{np.mean(ae)} - {np.max(ae)}')
                                ae = np.abs(np.squeeze(y_val)-y_val_pred)
                                print(f'{np.mean(ae)} - {np.max(ae)}')

                                print(f'\nmse (mean - max) | train / val')
                                mse_train = mean_squared_error(np.squeeze(y_train), y_train_pred)
                                print(f'{np.mean(mse_train)} - {np.max(mse_train)}')
                                mse_val = mean_squared_error(np.squeeze(y_val), y_val_pred)
                                print(f'{np.mean(mse_val)} - {np.max(mse_val)}')

                                print(f'\nrmse (mean - max) | train / val')
                                rmse = np.sqrt(mse_train)
                                print(f'{np.mean(rmse)} - {np.max(rmse)}')
                                rmse = np.sqrt(mse_val)
                                print(f'{np.mean(rmse)} - {np.max(rmse)}')

                                print(f'\nr2 (mean - max) | train / val')
                                r2 = r2_score(np.squeeze(y_train), y_train_pred)
                                print(f'{np.mean(r2)} - {np.max(r2)}')       
                                r2 = r2_score(np.squeeze(y_val), y_val_pred)
                                print(f'{np.mean(r2)} - {np.max(r2)}') 
                                
                                print('\n')