import argparse
import neptune
from getpass import getpass 

import dataset
import analysis


class Neptune:

        def __init__(self):
                self.PARAMS = {}


        def initialize(self, args):
                """
                Info
                """

                #self.PARAMS['parameter-one'] = args.parameter_one
                neptune.init(api_token=args.neptune_token, project_qualified_name=args.neptune_project_name)

                self.experiment_name = args.experiment_name 
                

        def create_experiment(self, args):
                """
                Info
                """

                self.initialize(args)

                self.experiment = neptune.create_experiment(self.experiment_name, params=self.PARAMS, upload_source_files=['*.py'])
                neptune.set_property('experiment-name', self.experiment_name)




def main(args):

        # Initialyze logging and tracking the data analysis/modeling experiment using Neptune
        log = Neptune()
        log.create_experiment(args)

        # Create and load train and test datasets
        train_dataset = dataset.Dataset(type='train')
        test_dataset = dataset.Dataset(type='test')
        train_dataset.load_dataset_from_csv()
        test_dataset.load_dataset_from_csv()

        # Select settings and sensors that matters
        selected_settings = ['setting1', 'setting2']
        selected_sensors = [s for s in train_dataset.sensors_head if s not in ['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']]
        
        # Get values of settings and sensors for a specific asset
        train_row_settings, train_row_sensors = train_dataset.get_settings_and_sensors_for_asset(asset_id=1, 
                                                                                                 cycle=1, 
                                                                                                 selected_settings=selected_settings, 
                                                                                                 selected_sensors=selected_sensors)
      
        test_row_settings, test_row_sensors = test_dataset.get_settings_and_sensors_for_asset(asset_id=1, 
                                                                                              cycle=1, 
                                                                                              selected_settings=selected_settings, 
                                                                                              selected_sensors=selected_sensors)

        # Get last cycle for a specific asset
        # For the train dataset, it is the failure cycle
        train_last_cycle_for_asset = train_dataset.get_asset_last_cycle(asset_id=1)
        test_last_cycle_for_asset = test_dataset.get_asset_last_cycle(asset_id=1)


        # Create train and test analysis
        train_analysis = analysis.Analysis(train_dataset) 
        test_analysis = analysis.Analysis(test_dataset)

        # Get the number of assets in each analysis (based on the respective dataset)
        train_assets_quantity = train_analysis.get_assets_quantity()
        test_assets_quantity = test_analysis.get_assets_quantity()

        # Get the array containing the last cycle of each asset
        train_assets_last_cycles_array = train_analysis.get_assets_last_cycle_array()
        test_assets_last_cycles_array = test_analysis.get_assets_last_cycle_array()

        #
        dummy_mean_precision = train_analysis.get_dummy_mean_precision(type='mean')
        print(f'dummy_mean_precision: {dummy_mean_precision}')
       
        # 
        train_analysis.log_violinchart(train_assets_last_cycles_array, log_category='target-charts', plot_name='failure-cycles')
        train_analysis.log_boxchart(train_assets_last_cycles_array, log_category='target-charts', plot_name='failure-cycles')
        train_analysis.compute_assets_last_cycle_statistics()

        #
        for ss in selected_settings:
                train_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)

        for ss in selected_sensors:
                train_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
                
                train_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)
       
                sensor_failure_values_array = train_analysis.get_sensors_last_value_for_assets(sensor_name=ss)
                train_analysis.log_violinchart(sensor_failure_values_array[:, 1], log_category='features-charts', plot_name=f'{ss}-failure-assets')
                train_analysis.log_boxchart(sensor_failure_values_array[:, 1], log_category='features-charts', plot_name=f'{ss}-failure-assets')




class Args:

        def __init__(self):
                self.neptune_token = getpass('Neptune token:')
                self.neptune_project_name = 'vitorvilela/suzano'
                self.experiment_name = 'dev'


if __name__ == '__main__':

        #parser = argparse.ArgumentParser(description='Predictive maintenance prediction.')

        #parser.add_argument('--neptune-token', dest='neptune_token', type=str, nargs='?', help='Neptune token to access you account and run the experiment.')
        #parser.add_argument('--neptune-project-name', dest='neptune_project_name', type=str, nargs='?', help='Neptune qualified project name.')
        #parser.add_argument('--neptune-experiment-name', dest='neptune_experiment_name', type=str, nargs='?', help='Neptune experiment name.')

        #args = parser.parse_args()
        args = Args()
        
        main(args)       