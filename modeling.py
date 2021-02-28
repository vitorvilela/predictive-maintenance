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

                self.numeric_features_name = ['monitoring-cycle'] + self.transformed_dataset.selected_settings + self.transformed_dataset.selected_sensors + self.transformed_dataset.selected_sensors_time_derivative
                self.features_name = self.numeric_features_name
                                
                self.set_targets_features_array()


        def set_targets_features_array(self):
                """                
                Returns: a tuple
                """

                self.X = self.transformed_dataset.dataframe[self.numeric_features_name]
                self.Y = self.transformed_dataset.dataframe[self.targets_name]


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
                        print('On get_pipeline')
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

                self.selected_models = [('DUMMY', DummyRegressor()), ('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()), ('KNN', KNeighborsRegressor(n_neighbors=5)), ('CART', DecisionTreeRegressor()), ('SVR', SVR()), ('BSN', BayesianRidge(compute_score=True)), ('AB', AdaBoostRegressor()), ('GB', GradientBoostingRegressor()), ('RF', RandomForestRegressor()), ('ET', ExtraTreesRegressor())]

                self.models_pipeline = []

                for model_name, model in self.selected_models:
                        self.models_pipeline.append((model_name, Pipeline([('data_pipeline', self.data_preprocess_pipeline.pipeline), (model_name, model)])))
                
                # # Simpler models
                # self.model_pipeline.append(('DUMMY', Pipeline([('data_preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('DUMMY', DummyRegressor())])))
                # self.model_pipeline.append(('LR', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('LR', LinearRegression())])))
                # self.model_pipeline.append(('LASSO', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('LASSO', Lasso())])))
                # self.model_pipeline.append(('EN', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('EN', ElasticNet())])))
                # self.model_pipeline.append(('KNN', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('KNN', KNeighborsRegressor())])))
                # self.model_pipeline.append(('CART', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('CART', DecisionTreeRegressor())])))
                # self.model_pipeline.append(('SVR', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('SVR', SVR())])))
                # self.model_pipeline.append(('BSN', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('BSN', BayesianRidge(compute_score=True))])))

                # # Boosting models
                # self.model_pipeline.append(('AB', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('AB', AdaBoostRegressor())])))
                # self.model_pipeline.append(('GBM', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('GBM', GradientBoostingRegressor())])))
      
                # # Bagging models
                # self.model_pipeline.append(('RF', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('RF', RandomForestRegressor())])))
                # self.model_pipeline.append(('ET', Pipeline([('preprocess_pipeline', self.data_preprocessing_pipeline.data_pipeline), ('ET', ExtraTreesRegressor())])))




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

                X = self.data_preprocess_pipeline.X
                Y = self.data_preprocess_pipeline.Y

                for degree in range(1, self.args.polynomial_features_degree+1):

                        print(f'\nDegree: {degree}')

                        self.data_preprocess_pipeline.set_polynomial_features(degree)
                        self.data_preprocess_pipeline.set_pipeline()
                        self.modeling_pipeline.set_pipeline()

                        for model_name, pipe in self.modeling_pipeline.models_pipeline:   

                                pipe.fit(X, Y)
                                
                                Y_pred = pipe.predict(X)

                                # Temporary model scores 
                                print(f'\nModel: {model_name}')
                                
                                print(f'\nape (mean - max)')
                                ape = np.abs(100*(np.squeeze(Y.values)-Y_pred)/np.squeeze(Y.values))
                                print(f'{np.mean(ape)} - {np.max(ape)}')
                                
                                print(f'\nae (mean - max)')
                                ae = np.abs(np.squeeze(Y.values)-Y_pred)
                                print(f'{np.mean(ae)} - {np.max(ae)}')

                                print(f'\nmse (mean - max)')
                                mse = mean_squared_error(np.squeeze(Y.values), Y_pred)
                                print(f'{np.mean(mse)} - {np.max(mse)}')

                                print(f'\nrmse (mean - max)')
                                rmse = np.sqrt(mse)
                                print(f'{np.mean(rmse)} - {np.max(rmse)}')

                                print(f'\nr2 (mean - max)')
                                r2 = r2_score(np.squeeze(Y.values), Y_pred)
                                print(f'{np.mean(r2)} - {np.max(r2)}')       
                                
                                print('\n')

                                


