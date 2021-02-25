import sys
from random import randrange
import numpy as np
import pandas as pd
import os
from neptunecontrib.api.table import log_table



class Dataset:

        cwd = os.getcwd()

        settings_head = ['setting{}'.format(s) for s in range(1, 4)]
        sensors_head = ['sensor{}'.format(s) for s in range(1, 22)]

        # It is used both in the csv file and dataframe
        # The train and test dataset files have the same header
        dataset_header = ['asset_id', 'cycle'] + settings_head + sensors_head


        def __init__(self, type):
                """
                Info
                """

                self.type = type
                csv_file_name = 'PM_train.csv' if type=='train' else 'PM_test.csv'                
                self.load_dataset_from_csv(csv_file_name)

                self.compute_assets_last_cycle_statistics()


        def load_dataset_from_csv(self, csv_file_name):
                """
                Info
                """

                cls = self.__class__

                csv_file_path = os.path.join(os.path.join(self.cwd, 'dataset'), csv_file_name)
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
                        print(f'In get_asset_dataframe(), there is not the column ({asset_id}) in the dataframe.')
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
                        print(f'In get_cycle_feature_array_for_asset(), there is not the column ({feature_name}) in the dataframe.')
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
                        print(f'In get_settings_and_sensors_for_asset(), there is not the columns ({asset_id}, {cycle}) in the dataset.')
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


        def compute_assets_last_cycle_statistics(self):
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

                log_table('assets-last-cycle-statistics', self.assets_last_cycle_descriptive_dataframe)   


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
                                print(f'In get_sensors_last_value_for_assets(), there is not the column ({sensor_name}) in the dataframe.')
                                sys.exit(1) 

                        sensor_last_values_list.append([asset, sensor_array[-1]])

                return np.array(sensor_last_values_list)



class TransformedDataset:

        # Current working directory. In case we export/import transformed dataset csv files.
        cwd = os.getcwd()

        #
        origin_cycle_period = 1

        #
        filter_window_size = 10

        #
        monitoring_cycle_step = int(0.5*filter_window_size)

        # 
        n_monitoring_cycles_per_asset = 10


        def __init__(self, dataset, selected_settings_sensors_tuple):
                """
                origin_selected_head: a tuple containing (selected_settings, selected_sensors) from the origin dataset
                """

                self.dataset = dataset
                self.type = dataset.type

                self.selected_settings = selected_settings_sensors_tuple[0]
                self.selected_sensors = selected_settings_sensors_tuple[1]
                self.selected_sensors_time_derivative = ['d{}dt'.format(s) for s in self.selected_sensors]

                # It is used both in the csv file and dataframe
                self.dataset_header_dict = {'train': ['rul', 'data-id', 'monitoring-cycle'] + self.selected_settings + self.selected_sensors + self.selected_sensors_time_derivative,
                                            'test': ['data-id', 'monitoring-cycle'] + self.selected_settings + self.selected_sensors + self.selected_sensors_time_derivative}

                self.dataset_header = self.dataset_header_dict[self.type]

                self.compute_min_monitoring_cycle()

                self.create_dataframe()

            

        def compute_min_monitoring_cycle(self):
                """
                Info
                """
                
                cls = self.__class__

                self.min_monitoring_cycle = self.dataset.assets_last_cycle_dict['min'] - 2*cls.filter_window_size


        def get_max_monitoring_cycle_for_asset(self, asset_id):
                """
                Info
                """

                max_monitoring_cycle = self.dataset.get_asset_last_cycle(asset_id)

                return max_monitoring_cycle        

            
        def pick_monitoring_cycle(self, asset_id, type='random'):
                """
                Info
                """

                cls = self.__class__

                max_monitoring_cycle = self.get_max_monitoring_cycle_for_asset(asset_id)
                
                monitoring_cycle = 0

                if type=='random':
                        monitoring_cycle = randrange(self.min_monitoring_cycle, max_monitoring_cycle, cls.monitoring_cycle_step)
                elif type=='fully':
                        monitoring_cycle = range(1, max_monitoring_cycle, cls.monitoring_cycle_step)
                
                yield monitoring_cycle


        def get_remaining_useful_life_value(self, asset_id, current_monitoring_cycle):
                """
                Info
                """

                max_monitoring_cycle = self.get_max_monitoring_cycle_for_asset(asset_id)
                rul = max_monitoring_cycle - current_monitoring_cycle

                return rul


        def filter_windowed_sensor_signal(self, windowed_sensor_signal, signal_filter='mean'):
                """
                Returns: a numpy array
                """

                filtered_signal = 0.

                if signal_filter=='mean':
                        filtered_signal = np.mean(windowed_sensor_signal)
                elif self.type=='low':
                        filtered_signal = windowed_sensor_signal[0]
                elif self.type=='high':
                        filtered_signal = windowed_sensor_signal[-1]
                else:
                        print(f'In filter_windowed_sensor_signal(), the signal_filter={signal_filter} is not available. Please use \'mean\', \'low\' or \'high\'.')
                        sys.exit(1)

                return filtered_signal


        def get_monitoring_sensor_and_derivative_values(self, asset_id, sensor_name, current_monitoring_cycle):
                """
                Returns: a float tuple
                """

                cls = self.__class__        

                cycle_sensor_array = self.dataset.get_cycle_feature_array_for_asset(asset_id, sensor_name)
                sensor_array = cycle_sensor_array[:, 1]

                low_index = current_monitoring_cycle - 2*cls.filter_window_size + 1
                mid_index = current_monitoring_cycle - cls.filter_window_size + 1
                high_index = current_monitoring_cycle

                present_windowed_sensor_signal = sensor_array[mid_index:high_index]
                past_windowed_sensor_signal = sensor_array[low_index:mid_index-1]

                present_signal_value = self.filter_windowed_sensor_signal(present_windowed_sensor_signal)
                past_signal_value = self.filter_windowed_sensor_signal(past_windowed_sensor_signal)
                signal_time_derivative = self.get_signal_time_derivative(present_signal_value, past_signal_value)

                return (present_signal_value, signal_time_derivative)


        def get_signal_time_derivative(self, present_signal_value, past_signal_value):
                """
                Info
                """

                cls = self.__class__ 

                signal_time_derivative = (present_signal_value - past_signal_value) / cls.filter_window_size

                return signal_time_derivative


        def get_monitoring_setting_value(self, asset_id, setting_name, current_monitoring_cycle):
                """
                Info
                """

                cycle_setting_array = self.dataset.get_cycle_feature_array_for_asset(asset_id, setting_name)
                setting_array = cycle_setting_array[:, 1]

                monitoring_setting = setting_array[current_monitoring_cycle]

                return monitoring_setting


        def create_dataframe(self):
                """
                Info
                """

                cls = self.__class__

                dataframe_rows = []

                data_id = 1

                # loop over assets
                for asset in self.dataset.assets:

                        n_monitoring_cycles_per_asset = cls.n_monitoring_cycles_per_asset

                        # loop over monitoring cycles
                        for _ in range(cls.n_monitoring_cycles_per_asset):
                               
                                current_monitoring_cycle = next(self.pick_monitoring_cycle(asset_id=asset, type='random'))
                                
                                rul = self.get_remaining_useful_life_value(asset_id=asset, current_monitoring_cycle=current_monitoring_cycle)

                                setting_row = []
                                sensor_and_time_derivative_row = []

                                monitoring_cycle_range_for_asset = self.get_max_monitoring_cycle_for_asset(asset_id=asset) - self.min_monitoring_cycle
                                monitoring_cycle_resolution_for_asset = int(monitoring_cycle_range_for_asset / cls.monitoring_cycle_step)

                                # If greater than 1.0 it would repeate data
                                coverage_monitoring_cycle_space = n_monitoring_cycles_per_asset / monitoring_cycle_resolution_for_asset                                
                                if coverage_monitoring_cycle_space > 1:
                                        n_monitoring_cycles_per_asset = monitoring_cycle_resolution_for_asset 
                                
                                # loop over settings
                                for setting in self.selected_settings:
                                        s = self.get_monitoring_setting_value(asset_id=asset, setting_name=setting, current_monitoring_cycle=current_monitoring_cycle)
                                        setting_row.append(s)
                                        
                                # loop sensors
                                for sensor in self.selected_sensors:
                                        s, dsdt = self.get_monitoring_sensor_and_derivative_values(asset_id=asset, sensor_name=sensor, current_monitoring_cycle=current_monitoring_cycle)
                                        sensor_and_time_derivative_row.append(s)
                                        sensor_and_time_derivative_row.append(dsdt)
      
                                if self.dataset.type=='train':
                                        dataframe_rows.append([rul] + [data_id] + [current_monitoring_cycle] + setting_row + sensor_and_time_derivative_row)
                                elif self.dataset.type=='test':        
                                        dataframe_rows.append([data_id] + [current_monitoring_cycle] + setting_row + sensor_and_time_derivative_row)
                                else:
                                        print(f'In create_dataframe(), the type={type} is not available. Please use \'train\' or \'test\'.')
                                        sys.exit(1)  

                                data_id += 1
                                   
                
                self.dataframe = pd.DataFrame(np.array(dataframe_rows),
                                              columns=self.dataset_header)                             