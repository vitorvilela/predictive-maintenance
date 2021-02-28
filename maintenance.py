import argparse

import neptune_settings as ns
import dataset
import analysis
import modeling




def main(args):

        # Initialyze logging and tracking the data analysis/modeling experiment using Neptune
        log = ns.Neptune(args)
        
        # Create train and test datasets
        train_dataset = dataset.Dataset(log=log, type='train')
        test_dataset = dataset.Dataset(log=log, type='test')


        # Select settings and sensors that matters
        selected_settings = ['setting1', 'setting2']
        selected_sensors = [s for s in train_dataset.sensors_head if s not in ['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']]


        # # Create train and test analysis
        train_dataset_analysis = analysis.DatasetAnalysis(log=log, dataset=train_dataset) 
        test_dataset_analysis = analysis.DatasetAnalysis(log=log, dataset=test_dataset)

        #
        dummy_type_list = [('min', 'mean'), ('min', 'max'), ('mean', 'mean'), ('mean', 'max')]
        for b, s in dummy_type_list:
                dummy_error = train_dataset_analysis.get_dummy_error(based_on=b, stats=s)
                log.exp.log_metric(f'dummy {s} error based on {b}', dummy_error)
                dummy_percentage_error = train_dataset_analysis.get_dummy_percentage_error(based_on=b, stats=s)
                log.exp.log_metric(f'dummy {s} percentage error based on {b}', dummy_percentage_error)
        
        # Get the array containing the last cycle of each asset
        train_assets_last_cycles_array = train_dataset.get_assets_last_cycle_array()
        test_assets_last_cycles_array = test_dataset.get_assets_last_cycle_array()

        # 
        train_dataset.set_assets_last_cycle_statistics()
        train_dataset_analysis.log_violinchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='failure-cycle')
        train_dataset_analysis.log_boxchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='failure-cycle')
        
        #
        test_dataset.set_assets_last_cycle_statistics()
        test_dataset_analysis.log_violinchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='failure-cycle')
        test_dataset_analysis.log_boxchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='failure-cycle')


        #
        for ss in selected_settings:
                train_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        for ss in selected_sensors:
                train_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
                train_dataset_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
                train_sensor_failure_values_array = train_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
                train_dataset_analysis.log_violinchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'{ss}-failure-assets')
                train_dataset_analysis.log_boxchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'{ss}-failure-assets')

        #
        for ss in selected_settings:
                test_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        for ss in selected_sensors:
                test_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
                test_dataset_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
                test_sensor_failure_values_array = test_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
                test_dataset_analysis.log_violinchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'{ss}-failure-assets')
                test_dataset_analysis.log_boxchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'{ss}-failure-assets')


        # Create train and test transformed datasets
        train_transformed_dataset = dataset.TransformedDataset(log=log, dataset=train_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))
        print(f'train_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')
        print(train_transformed_dataset.dataframe)
        print(f'\ntrain_transformed_dataset.dataframe shape: {train_transformed_dataset.dataframe.shape}\n')

        test_transformed_dataset = dataset.TransformedDataset(log=log, dataset=test_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))  
        print(f'test_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')   
        print(test_transformed_dataset.dataframe)

        # Log dataframe info into stdout
        print(train_transformed_dataset.dataframe.dtypes)        
        print(train_transformed_dataset.dataframe.describe())


        # Create train transformed dataset analysis
        train_transformed_dataset_analysis = analysis.TransformedDatasetAnalysis(log=log, transformed_dataset=train_transformed_dataset)

        train_transformed_dataset_analysis.log_correlation_matrix(log_category='transformed-dataset-train-charts')

        
        feature_array = train_transformed_dataset.get_feature_array(feature_name='monitoring-cycle')
        train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='monitoring-cycle')
        train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='monitoring-cycle')
        train_transformed_dataset_analysis.log_scatterchart(feature_name='monitoring-cycle', log_category='transformed-dataset-train-charts')

        feature_array = train_transformed_dataset.get_feature_array(feature_name='rul')
        train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='rul')
        train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='rul')
        train_transformed_dataset_analysis.log_scatterchart(feature_name='rul', log_category='transformed-dataset-train-charts')

        for ss in selected_settings:
                feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
                train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')

        for ss in selected_sensors:
                feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
                train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')

        for ss in train_transformed_dataset.selected_sensors_time_derivative:        
                feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
                train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'{ss}')
                train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')


        # Modeling
        train_preprocess_pipeline = modeling.DataPreprocessPipeline(log=log, transformed_dataset=train_transformed_dataset)
        train_modeling_pipeline = modeling.ModelingPipeline(log=log, data_preprocess_pipeline=train_preprocess_pipeline)
        train_model = modeling.Model(log=log, modeling_pipeline=train_modeling_pipeline)











if __name__ == '__main__':


        # TODO Get experiment parameters through CLI to allow automate design of experiments and fine-tuning parameters
        #parser = argparse.ArgumentParser(description='Predictive maintenance prediction.')
        #parser.add_argument('--neptune-token', dest='neptune_token', type=str, nargs='?', help='Neptune token to access you account and run the experiment.')
        #parser.add_argument('--neptune-project-name', dest='neptune_project_name', type=str, nargs='?', help='Neptune qualified project name.')
        #parser.add_argument('--neptune-experiment-name', dest='neptune_experiment_name', type=str, nargs='?', help='Neptune experiment name.')
        #args = parser.parse_args()

        args = ns.Args()
        
        main(args)       