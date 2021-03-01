import sys
from random import randrange
import numpy as np
import pandas as pd
import os
from neptunecontrib.api.table import log_table

import neptune_settings as ns










class Dataset:

        cwd = os.getcwd()

        settings_head = ['setting{}'.format(s) for s in range(1, 4)]
        sensors_head = ['sensor{}'.format(s) for s in range(1, 22)]

        # It is used both in the csv file and dataframe
        # The train and test dataset files have the same header
        dataset_header = ['asset_id', 'cycle'] + settings_head + sensors_head


        def __init__(self, log, type):
                """
                Info
                """

                self.log = log
                self.args = log.args  
                self.type = type
                csv_file_name = 'PM_train.csv' if type=='train' else 'PM_test.csv'                
                self.load_dataset_from_csv(csv_file_name)

                self.set_assets_last_cycle_statistics()


        def load_dataset_from_csv(self, csv_file_name):
                """
                Info
                """

                cls = self.__class__

                csv_file_path = os.path.join(os.path.join(cls.cwd, 'dataset'), csv_file_name)
                self.dataframe = pd.read_csv(csv_file_path, names=cls.dataset_header, header=None, delim_whitespace=True)

                self.assets = np.unique(self.dataframe['asset_id'].values)
                self.n_assets = len(self.assets)


        def get_asset_dataframe(self, asset_id):
                """
                Returns: a pandas dataframe
                """

                df = self.dataframe

                try:
                        asset_dataframe = df.loc[(df['asset_id']==asset_id)]
                except:
                        print(f'\nIn get_asset_dataframe(), there is not the column ({asset_id}) in the dataframe.\n')
                        sys.exit(1) 

                return asset_dataframe


        def get_cycle_feature_array_for_asset(self, asset_id, feature_name):
                """
                Returns: a float, 2D numpy array
                """

                df = self.get_asset_dataframe(asset_id)
                
                try:
                        cycle_feature_array = df[['cycle', feature_name]].values
                except:
                        print(f'\nIn get_cycle_feature_array_for_asset(), there is not the column ({feature_name}) in the dataframe.\n')
                        sys.exit(1) 


                return cycle_feature_array


        def get_settings_and_sensors_for_asset_and_cycle(self, asset_id, cycle, selected_settings='from_header', selected_sensors='from_header'):
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
                        print(f'\nIn get_settings_and_sensors_for_asset(), there is not the columns ({asset_id}, {cycle}) in the dataset.\n')
                        sys.exit(1)

                return (np.squeeze(settings_array), np.squeeze(sensors_array))


        def get_assets_last_cycle_array(self):
                """
                Info
                """
          
                assets_last_cycle_array = np.array([self.get_asset_last_cycle(a) for a in self.assets])

                return assets_last_cycle_array        


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


        def set_assets_last_cycle_statistics(self):
                """
                Returns: a pandas dataframe
                """

                assets_last_cycle_array = self.get_assets_last_cycle_array()

                self.assets_last_cycle_dict = {}
                self.assets_last_cycle_dict['count'] = int(len(assets_last_cycle_array))
                self.assets_last_cycle_dict['min'] = int(np.min(assets_last_cycle_array))
                self.assets_last_cycle_dict['max'] = int(np.max(assets_last_cycle_array))
                self.assets_last_cycle_dict['mean'] = int(np.mean(assets_last_cycle_array))
                self.assets_last_cycle_dict['std'] = int(np.std(assets_last_cycle_array))
                self.assets_last_cycle_dict['p25'] = int(np.percentile(assets_last_cycle_array, q=25))
                self.assets_last_cycle_dict['median'] = int(np.median(assets_last_cycle_array))
                self.assets_last_cycle_dict['p75'] = int(np.percentile(assets_last_cycle_array, q=75))

                stats = [item[0] for item in self.assets_last_cycle_dict.items()]
                values = [item[1] for item in self.assets_last_cycle_dict.items()]
                self.assets_last_cycle_descriptive_dataframe = pd.DataFrame({'stats': stats, 'values': values})

                log_table(f'{self.type}-assets-last-cycle-statistics', self.assets_last_cycle_descriptive_dataframe)   


        def get_sensors_last_value_for_assets(self, sensor_name):
                """
                Info
                """

                sensor_last_values_list = []

                for asset in self.assets:

                        df = self.get_asset_dataframe(asset_id=asset)
                        
                        try:        
                                sensor_array = df[sensor_name].values
                        except:
                                print(f'\nIn get_sensors_last_value_for_assets(), there is not the column ({sensor_name}) in the dataframe.\n')
                                sys.exit(1) 

                        sensor_last_values_list.append([asset, sensor_array[-1]])

                return np.array(sensor_last_values_list)










class TransformedDataset:

        # Current working directory. In case we export/import transformed dataset csv files.
        cwd = os.getcwd()

        def __init__(self, log, dataset, selected_settings_sensors_tuple):
                """
                origin_selected_head: a tuple containing (selected_settings, selected_sensors) from the origin dataset
                """

                self.log = log
                self.args = log.args                
                self.dataset = dataset
                self.type = dataset.type

                self.selected_settings = selected_settings_sensors_tuple[0]
                self.selected_sensors = selected_settings_sensors_tuple[1]

                if self.args.option_use_derivatives:
                        self.selected_sensors_time_derivative = ['ds{}dt'.format(s.split('sensor')[1]) for s in self.selected_sensors]
                        self.selected_features = self.selected_settings + self.selected_sensors + self.selected_sensors_time_derivative
                else:
                        self.selected_features = self.selected_settings + self.selected_sensors

                # It is used both in the csv file and dataframe
                self.dataset_header_dict = {'train': ['data-id', 'monitoring-cycle'] + self.selected_features + ['rul'],
                                            'test': ['data-id', 'monitoring-cycle'] + self.selected_features}

                self.dataset_header = self.dataset_header_dict[self.type]

                self.set_min_monitoring_cycle()

                self.set_dataframe()


        def get_feature_array(self, feature_name):
                """
                Returns: a numpy array
                """
                
                try:
                        feature_array = self.dataframe[[feature_name]].values
                except:
                        print(f'\nIn get_feature_array(), there is not the column ({feature_name}) in the dataframe.\n')
                        sys.exit(1) 

                return feature_array        
            

        def set_min_monitoring_cycle(self):
                """
                Info
                """
                             
                if not self.args.option_min_monitoring_cycle_constant:                
                        self.min_monitoring_cycle = self.dataset.assets_last_cycle_dict['min'] - self.args.filter_window_size
                else:
                        self.min_monitoring_cycle = self.args.min_monitoring_cycle_constant

                if self.min_monitoring_cycle - 2*self.args.filter_window_size < 1:
                        print(f'\nIn set_min_monitoring_cycle(), min_monitoring_cycle lesser than allowed value (1)\n')
                        sys.exit(1)


        def get_max_monitoring_cycle_for_asset(self, asset_id):
                """
                Info
                """

                max_monitoring_cycle = self.dataset.get_asset_last_cycle(asset_id)

                return max_monitoring_cycle        

            
        def pick_monitoring_cycle(self, asset_id, pick_type='random'):
                """
                Info
                """

                max_monitoring_cycle = self.get_max_monitoring_cycle_for_asset(asset_id)

                if pick_type=='random':                        
                        return randrange(self.min_monitoring_cycle, max_monitoring_cycle, self.args.monitoring_cycle_step)
                else:
                        print(f'\nIn pick_monitoring_cycle(), the pick_type={pick_type} is not available. Please use \'random\'.\n')
                        sys.exit(1)                
                        

        def get_remaining_useful_life_value(self, asset_id, current_monitoring_cycle):
                """
                Info
                """

                max_monitoring_cycle = self.get_max_monitoring_cycle_for_asset(asset_id)
                rul = max_monitoring_cycle - current_monitoring_cycle

                return rul


        def filter_windowed_sensor_signal(self, windowed_sensor_signal, filter_type='mean'):
                """
                Returns: a numpy array
                """

                filtered_signal = 0.

                if filter_type=='mean':
                        filtered_signal = np.mean(windowed_sensor_signal)
                elif filter_type=='low':
                        filtered_signal = windowed_sensor_signal[0]
                elif filter_type=='high':
                        filtered_signal = windowed_sensor_signal[-1]
                else:
                        print(f'\nIn filter_windowed_sensor_signal(), the filter_type={filter_type} is not available. Please use \'mean\', \'low\' or \'high\'.\n')
                        sys.exit(1)

                return filtered_signal


        def get_monitoring_sensor_and_derivative_values(self, asset_id, sensor_name, current_monitoring_cycle):
                """
                Returns: a float tuple
                """

                cycle_sensor_array = self.dataset.get_cycle_feature_array_for_asset(asset_id, sensor_name)
                sensor_array = cycle_sensor_array[:, 1]

                low_index = (current_monitoring_cycle - 2*self.args.filter_window_size + 1) - 1 
                if low_index < 1: 
                        print(f'\nIn get_monitoring_sensor_and_derivative_values(), low_index < 1\n')
                        sys.exit(1)
                mid_index = (current_monitoring_cycle - self.args.filter_window_size + 1) - 1 
                high_index = current_monitoring_cycle - 1

                
                present_windowed_sensor_signal = sensor_array[mid_index:high_index]
                present_signal_value = self.filter_windowed_sensor_signal(present_windowed_sensor_signal)

                signal_time_derivative = 0.

                if self.args.option_use_derivatives:
                        past_windowed_sensor_signal = sensor_array[low_index:mid_index-1]                
                        past_signal_value = self.filter_windowed_sensor_signal(past_windowed_sensor_signal)
                        signal_time_derivative = self.get_signal_time_derivative(present_signal_value, past_signal_value)

                        return (present_signal_value, signal_time_derivative)

                else:
                        return (present_signal_value, None)        


        def get_signal_time_derivative(self, present_signal_value, past_signal_value):
                """
                Info
                """

                signal_time_derivative = (present_signal_value - past_signal_value) / self.args.filter_window_size

                return signal_time_derivative


        def get_monitoring_setting_value(self, asset_id, setting_name, current_monitoring_cycle):
                """
                Info
                """

                cycle_setting_array = self.dataset.get_cycle_feature_array_for_asset(asset_id, setting_name)
                setting_array = cycle_setting_array[:, 1]

                monitoring_setting = setting_array[current_monitoring_cycle-1]

                return monitoring_setting


        def get_n_monitoring_cycle_for_asset(self, asset):
                """
                Info
                """

                # The restraint factor is an approach to handle repeated monitoring_cycle in the random function
                coverage_restraint = self.args.coverage_restraint

                n_monitoring_cycles_for_asset = 0

                if self.dataset.type=='train':                           

                        n_monitoring_cycles_for_asset = self.args.n_monitoring_cycles_per_asset

                        monitoring_cycle_range_for_asset = self.get_max_monitoring_cycle_for_asset(asset_id=asset) - self.min_monitoring_cycle
                        monitoring_cycle_resolution_for_asset = int(monitoring_cycle_range_for_asset / self.args.monitoring_cycle_step)

                        # If coverage is greater than a threshold near 1.0 (e.g., 0.5), it would mistakenly repeate data into the transformed dataset
                        coverage_monitoring_cycle_space = n_monitoring_cycles_for_asset / monitoring_cycle_resolution_for_asset   
                        #print(f'asset {asset}: coverage_monitoring_cycle_space ({coverage_monitoring_cycle_space})')
                                
                        if coverage_monitoring_cycle_space > coverage_restraint:                                
                                n_monitoring_cycles_for_asset = int(coverage_restraint*monitoring_cycle_resolution_for_asset) 
                                coverage_monitoring_cycle_space = n_monitoring_cycles_for_asset / monitoring_cycle_resolution_for_asset   
                                #print(f'\tcorrected coverage_monitoring_cycle_space ({coverage_monitoring_cycle_space})')

                elif self.dataset.type=='test':

                        n_monitoring_cycles_for_asset = 1

                else:
                        print(f'\nIn create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.\n')
                        sys.exit(1)               

                return n_monitoring_cycles_for_asset



        def get_transformed_dataframe_size(self):
                """
                Info
                """

                transformed_dataframe_size = 0

                for asset in self.dataset.assets:
                        transformed_dataframe_size += self.get_n_monitoring_cycle_for_asset(asset)

                return transformed_dataframe_size



        def set_dataframe(self):
                """
                Info
                """                

                data_id_rows = []
                monitoring_cycle_rows = []
                rul_rows = []
                features_dict = {}
                                
                current_monitoring_cycle = 0
                data_id = 0


                if self.dataset.type=='train':

                        transformed_dataframe_size = self.get_transformed_dataframe_size()
                        #print(f'\ntransformed_dataframe_size: {transformed_dataframe_size}\n')
                        
                        for selected_feature in self.selected_features:                        
                                features_dict[selected_feature] = [1.e10]*transformed_dataframe_size   

                elif self.dataset.type=='test':

                        for selected_feature in self.selected_features:                        
                                features_dict[selected_feature] = [1.e10]*self.dataset.assets

                else:
                        print(f'\nIn create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.\n')
                        sys.exit(1)


                # Loop over assets to create dataframe
                for asset in self.dataset.assets:

                                  
                        n_monitoring_cycles_for_asset = self.get_n_monitoring_cycle_for_asset(asset)
                        
                        # Loop over monitoring cycles
                        for _ in range(n_monitoring_cycles_for_asset):
                               
                                if self.dataset.type=='train':

                                        current_monitoring_cycle = self.pick_monitoring_cycle(asset_id=asset, pick_type='random')
                                        rul = self.get_remaining_useful_life_value(asset_id=asset, current_monitoring_cycle=current_monitoring_cycle)
                                        rul_rows.append(int(rul))
                                
                                elif self.dataset.type=='test':  

                                        current_monitoring_cycle = self.get_max_monitoring_cycle_for_asset(asset_id=asset)
                                
                                else:

                                        print(f'\nIn create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.\n')
                                        sys.exit(1)   

                                monitoring_cycle_rows.append(int(current_monitoring_cycle))                                     
                                                                
                                # Loop over settings
                                for setting in self.selected_settings:
                                        s = self.get_monitoring_setting_value(asset_id=asset, setting_name=setting, current_monitoring_cycle=current_monitoring_cycle)                                       
                                        features_dict[setting][data_id] = s
                                                                        
                                # loop sensors
                                if self.args.option_use_derivatives:
                                        for sensor, sensor_time_derivative in zip(self.selected_sensors, self.selected_sensors_time_derivative):
                                                s, dsdt = self.get_monitoring_sensor_and_derivative_values(asset_id=asset, sensor_name=sensor, current_monitoring_cycle=current_monitoring_cycle)
                                                features_dict[sensor][data_id] = s
                                                features_dict[sensor_time_derivative][data_id] = dsdt
                                else:
                                        for sensor in self.selected_sensors:
                                                s, _ = self.get_monitoring_sensor_and_derivative_values(asset_id=asset, sensor_name=sensor, current_monitoring_cycle=current_monitoring_cycle)
                                                features_dict[sensor][data_id] = s


                                data_id += 1
                                data_id_rows.append(int(data_id))

                                # Break loop over monitoring cycles
                                if self.dataset.type=='test': 
                                        break        


                dataframe_dict = {'data-id': data_id_rows, 'monitoring-cycle': monitoring_cycle_rows}

                if self.dataset.type=='train':
                        dataframe_dict['rul'] = rul_rows  
                elif self.dataset.type=='test':
                        pass     
                else:
                        print(f'\nIn create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.\n')
                        sys.exit(1)   


                for feature in self.selected_features:
                        dataframe_dict[feature] = features_dict[feature]

                self.dataframe = pd.DataFrame(data=dataframe_dict)   