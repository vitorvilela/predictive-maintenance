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


        # Create train and test datasets
        train_dataset = dataset.Dataset(type='train')
        test_dataset = dataset.Dataset(type='test')


        # Select settings and sensors that matters
        selected_settings = ['setting1', 'setting2']
        selected_sensors = [s for s in train_dataset.sensors_head if s not in ['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']]


        # Get values of settings and sensors for a specific asset
        train_row_settings, train_row_sensors = train_dataset.get_settings_and_sensors_for_asset_and_cycle(asset_id=1, 
                                                                                                           cycle=1, 
                                                                                                           selected_settings=selected_settings, 
                                                                                                           selected_sensors=selected_sensors)
      
        test_row_settings, test_row_sensors = test_dataset.get_settings_and_sensors_for_asset_and_cycle(asset_id=1, 
                                                                                                        cycle=1, 
                                                                                                        selected_settings=selected_settings, 
                                                                                                        selected_sensors=selected_sensors)


        # Get last cycle for a specific asset
        # For the train dataset, it is the failure cycle
        train_last_cycle_for_asset = train_dataset.get_asset_last_cycle(asset_id=1)
        test_last_cycle_for_asset = test_dataset.get_asset_last_cycle(asset_id=1)


        # Create train and test analysis
        train_analysis = analysis.DatasetAnalysis(train_dataset) 
        test_analysis = analysis.DatasetAnalysis(test_dataset)

        #
        dummy_mean_precision = train_analysis.get_dummy_mean_precision(type='mean')
        neptune.log_metric(f'dummy-mean-precision-type-mean', dummy_mean_precision)
        
        dummy_mean_precision = train_analysis.get_dummy_mean_precision(type='min')
        neptune.log_metric(f'dummy-mean-precision-type-min', dummy_mean_precision)
       

        # Get the array containing the last cycle of each asset
        train_assets_last_cycles_array = train_dataset.get_assets_last_cycle_array()
        test_assets_last_cycles_array = test_dataset.get_assets_last_cycle_array()

        # 
        train_dataset.compute_assets_last_cycle_statistics()
        train_analysis.log_violinchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='train-failure-cycles')
        train_analysis.log_boxchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='train-failure-cycles')
        
        #
        test_dataset.compute_assets_last_cycle_statistics()
        test_analysis.log_violinchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='test-failure-cycles')
        test_analysis.log_boxchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='test-failure-cycles')



        #
        for ss in selected_settings:
                train_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        for ss in selected_sensors:
                train_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
                train_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
                train_sensor_failure_values_array = train_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
                train_analysis.log_violinchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'train-{ss}-failure-assets')
                train_analysis.log_boxchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'train-{ss}-failure-assets')

        #
        for ss in selected_settings:
                test_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        for ss in selected_sensors:
                test_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
                test_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
                test_sensor_failure_values_array = test_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
                test_analysis.log_violinchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'test-{ss}-failure-assets')
                test_analysis.log_boxchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'test-{ss}-failure-assets')



        # Create train and test transformed datasets
        train_transformed_dataset = dataset.TransformedDataset(dataset=train_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))
        print(f'train_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')
        print(train_transformed_dataset.dataframe)

        test_transformed_dataset = dataset.TransformedDataset(dataset=test_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))  
        print(f'test_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')   
        print(test_transformed_dataset.dataframe)




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