import sys
import numpy as np
import pandas as pd
import os

class Dataset:

        cwd = os.getcwd()

        settings_head = ['setting{}'.format(s) for s in range(1,4)]
        sensors_head = ['sensor{}'.format(s) for s in range(1,22)]

        # Train and test dataset files have the same header
        csv_header = ['asset_id', 'cycle'] + settings_head + sensors_head

        def __init__(self, type):
                self.csv_file_name = 'PM_train.csv' if type=='train' else 'PM_test.csv'
                               

        def load_dataset_from_csv(self):
                """
                Info
                """

                cls = self.__class__

                csv_file_path = os.path.join(os.path.join(self.cwd, 'dataset'), self.csv_file_name)
                self.dataframe = pd.read_csv(csv_file_path, names=cls.csv_header, header=None, delim_whitespace=True)

                self.assets = np.unique(self.dataframe['asset_id'].values)


        def get_asset_dataframe(self, asset_id):
                """
                Returns: a pandas dataframe
                """

                df = self.dataframe
                asset_dataframe = df.loc[(df['asset_id']==asset_id)]

                return asset_dataframe


        def get_cycle_feature_array(self, asset_id, feature_name):
                """
                Returns: a float, 2D numpy array
                """

                df = self.get_asset_dataframe(asset_id)
                cycle_feature_array = df[['cycle', feature_name]].values

                return cycle_feature_array


        def get_settings_and_sensors_for_asset(self, asset_id, cycle, selected_settings='from_header', selected_sensors='from_header'):
                """
                What are the settings and sensors values given an asset_id, a cycle and a list of selected settings and sensors?
                Returns: a tuple of numpy array
                """
   
                settings_list = self.settings_head if selected_settings=='from_header' else selected_settings                
                sensors_list = self.sensors_head if selected_sensors=='from_header' else selected_sensors

                try:
                        df = self.get_asset_dataframe(asset_id)
                        settings_array = df.loc[df['cycle']==cycle][settings_list].values
                        sensors_array = df.loc[df['cycle']==cycle][sensors_list].values                        
                except:
                        print(f'In get_settings_and_sensors_for_asset(), there is not the input ({asset_id}, {cycle}) in the dataset.')
                        sys.exit(1)

                return (np.squeeze(settings_array), np.squeeze(sensors_array))


        def get_asset_last_cycle(self, asset_id):
                """
                What is the last cycle for a given asset_id? 
                For the train dataset it means the failure cycle
                Returns: an integer
                """

                df = self.get_asset_dataframe(asset_id)
                cycle_array = df['cycle'].values
                asset_last_cycle = np.max(cycle_array)

                return asset_last_cycle