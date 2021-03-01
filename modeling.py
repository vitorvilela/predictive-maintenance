import sys
import numpy as np

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



class DataPreprocessPipeline:

        def __init__(self, log, transformed_dataset):
                """
                Info
                """

                self.log = log
                self.args = log.args
                self.transformed_dataset = transformed_dataset

                #
                if self.transformed_dataset.type=='train':
                        self.targets_name = ['rul']
                else:
                        self.targets_name = None      

                if self.args.option_use_derivatives:
                        self.numeric_features_name = ['monitoring-cycle'] + self.transformed_dataset.selected_settings + self.transformed_dataset.selected_sensors + self.transformed_dataset.selected_sensors_time_derivative
                else: 
                        self.numeric_features_name = ['monitoring-cycle'] + self.transformed_dataset.selected_settings + self.transformed_dataset.selected_sensors
                
                self.features_name = self.numeric_features_name
                                
                self.set_targets_features_array()


        def set_targets_features_array(self):
                """                
                Returns: a tuple
                """

                self.train_val_split_seed = 1
                self.val_ratio = 0.3

                x = self.transformed_dataset.dataframe[self.numeric_features_name]
                y = self.transformed_dataset.dataframe[self.targets_name]     

                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=self.val_ratio, random_state=self.train_val_split_seed) 


        def set_polynomial_features(self, degree): 
                """
                Info
                """

                self.polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)


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



class ModelingPipeline:

        def __init__(self, log, data_preprocess_pipeline):
                """
                Info
                """

                self.log = log
                self.args = log.args
                self.data_preprocess_pipeline = data_preprocess_pipeline

                
        def set_pipeline(self):
                """
                Info
                """

                #self.selected_models = [('DUMMY', DummyRegressor()), ('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()), ('KNN', KNeighborsRegressor(n_neighbors=5)), ('CART', DecisionTreeRegressor()), ('SVR', SVR()), ('BSN', BayesianRidge(compute_score=True)), ('AB', AdaBoostRegressor()), ('GB', GradientBoostingRegressor()), ('RF', RandomForestRegressor()), ('ET', ExtraTreesRegressor())]
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

                self.get_models()


        def get_models(self):
                """
                Info
                """

                self.pipeline = []

                x_train = self.data_preprocess_pipeline.x_train
                x_val = self.data_preprocess_pipeline.x_val
                y_train = self.data_preprocess_pipeline.y_train
                y_val = self.data_preprocess_pipeline.y_val

                for degree in range(1, self.args.polynomial_features_degree+1):

                        print(f'\nDegree: {degree}')

                        self.data_preprocess_pipeline.set_polynomial_features(degree)
                        self.data_preprocess_pipeline.set_pipeline()
                        self.modeling_pipeline.set_pipeline()

                        for model_name, pipe in self.modeling_pipeline.pipeline:   

                                self.pipeline.append(pipe.fit(x_train, y_train))
                                
                                y_train_pred = pipe.predict(x_train)
                                y_val_pred = pipe.predict(x_val)
                                

                                print(f'\nModel: {model_name}')
                                
                                # print('pipe.score train subset')
                                # print(pipe.score(x_train, y_train))
                                # print('pipe.score val subset')
                                # print(pipe.score(x_val, y_val))                               

                                print(f'\nape (mean - max) | train / val')
                                ape = np.abs(100*(np.squeeze(y_train.values)-y_train_pred)/np.squeeze(y_train.values))
                                print(f'{np.mean(ape)} - {np.max(ape)}')
                                ape = np.abs(100*(np.squeeze(y_val.values)-y_val_pred)/np.squeeze(y_val.values))
                                print(f'{np.mean(ape)} - {np.max(ape)}')
                                
                                print(f'\nae (mean - max) | train / val')
                                ae = np.abs(np.squeeze(y_train.values)-y_train_pred)
                                print(f'{np.mean(ae)} - {np.max(ae)}')
                                ae = np.abs(np.squeeze(y_val.values)-y_val_pred)
                                print(f'{np.mean(ae)} - {np.max(ae)}')

                                print(f'\nmse (mean - max) | train / val')
                                mse_train = mean_squared_error(np.squeeze(y_train.values), y_train_pred)
                                print(f'{np.mean(mse_train)} - {np.max(mse_train)}')
                                mse_val = mean_squared_error(np.squeeze(y_val.values), y_val_pred)
                                print(f'{np.mean(mse_val)} - {np.max(mse_val)}')

                                print(f'\nrmse (mean - max) | train / val')
                                rmse = np.sqrt(mse_train)
                                print(f'{np.mean(rmse)} - {np.max(rmse)}')
                                rmse = np.sqrt(mse_val)
                                print(f'{np.mean(rmse)} - {np.max(rmse)}')

                                print(f'\nr2 (mean - max) | train / val')
                                r2 = r2_score(np.squeeze(y_train.values), y_train_pred)
                                print(f'{np.mean(r2)} - {np.max(r2)}')       
                                r2 = r2_score(np.squeeze(y_val.values), y_val_pred)
                                print(f'{np.mean(r2)} - {np.max(r2)}') 
                                
                                print('\n')

                                


