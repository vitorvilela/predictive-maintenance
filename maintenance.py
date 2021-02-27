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


        # Get values of settings and sensors for a specific asset
        #train_row_settings, train_row_sensors = train_dataset.get_settings_and_sensors_for_asset_and_cycle(asset_id=1, 
        #                                                                                                   cycle=1, 
        #                                                                                                   selected_settings=selected_settings, 
        #                                                                                                   selected_sensors=selected_sensors)
      
        #test_row_settings, test_row_sensors = test_dataset.get_settings_and_sensors_for_asset_and_cycle(asset_id=1, 
        #                                                                                                cycle=1, 
        #                                                                                                selected_settings=selected_settings, 
        #                                                                                                selected_sensors=selected_sensors)


        # Get last cycle for a specific asset
        # For the train dataset, it is the failure cycle
        #train_last_cycle_for_asset = train_dataset.get_asset_last_cycle(asset_id=1)
        #test_last_cycle_for_asset = test_dataset.get_asset_last_cycle(asset_id=1)


        # # Create train and test analysis
        # train_dataset_analysis = analysis.DatasetAnalysis(log=log, dataset=train_dataset) 
        # test_dataset_analysis = analysis.DatasetAnalysis(log=log, dataset=test_dataset)

        # #
        # dummy_mean_precision = train_dataset_analysis.get_dummy_mean_precision(type='mean')
        # log.exp.log_metric(f'dummy-mean-precision-type-mean', dummy_mean_precision)
        
        # dummy_mean_precision = train_dataset_analysis.get_dummy_mean_precision(type='min')
        # log.exp.log_metric(f'dummy-mean-precision-type-min', dummy_mean_precision)
       

        # # Get the array containing the last cycle of each asset
        # train_assets_last_cycles_array = train_dataset.get_assets_last_cycle_array()
        # test_assets_last_cycles_array = test_dataset.get_assets_last_cycle_array()

        # # 
        # train_dataset.compute_assets_last_cycle_statistics()
        # train_dataset_analysis.log_violinchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='train-failure-cycles')
        # train_dataset_analysis.log_boxchart(train_assets_last_cycles_array, log_category='train-target-charts', plot_name='train-failure-cycles')
        
        # #
        # test_dataset.compute_assets_last_cycle_statistics()
        # test_dataset_analysis.log_violinchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='test-failure-cycles')
        # test_dataset_analysis.log_boxchart(test_assets_last_cycles_array, log_category='test-target-charts', plot_name='test-failure-cycles')



        # #
        # for ss in selected_settings:
        #         train_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        # for ss in selected_sensors:
        #         train_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
        #         train_dataset_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
        #         train_sensor_failure_values_array = train_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
        #         train_dataset_analysis.log_violinchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'train-{ss}-failure-assets')
        #         train_dataset_analysis.log_boxchart(train_sensor_failure_values_array[:, 1], log_category='train-features-charts', plot_name=f'train-{ss}-failure-assets')

        # #
        # for ss in selected_settings:
        #         test_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)
        # for ss in selected_sensors:
        #         test_dataset_analysis.log_feature_linechart_for_asset(asset_id=75, feature_name=ss)                
        #         test_dataset_analysis.log_sensor_failure_value_linechart_for_assets(sensor_name=ss)       
        #         test_sensor_failure_values_array = test_dataset.get_sensors_last_value_for_assets(sensor_name=ss)
        #         test_dataset_analysis.log_violinchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'test-{ss}-failure-assets')
        #         test_dataset_analysis.log_boxchart(test_sensor_failure_values_array[:, 1], log_category='test-features-charts', plot_name=f'test-{ss}-failure-assets')



        # Create train and test transformed datasets
        train_transformed_dataset = dataset.TransformedDataset(log=log, dataset=train_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))
        print(f'train_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')
        print(train_transformed_dataset.dataframe)

        test_transformed_dataset = dataset.TransformedDataset(log=log, dataset=test_dataset, selected_settings_sensors_tuple=(selected_settings, selected_sensors))  
        print(f'test_dataset.dataframe has nan: {train_transformed_dataset.dataframe.isnull().values.any()}')   
        print(test_transformed_dataset.dataframe)

        # Log dataframe info into stdout
        print(train_transformed_dataset.dataframe.dtypes)        
        print(train_transformed_dataset.dataframe.describe())



        ## Create train transformed dataset analysis
        #train_transformed_dataset_analysis = analysis.TransformedDatasetAnalysis(log=log, transformed_dataset=train_transformed_dataset)

        ##
        # train_transformed_dataset_analysis.log_correlation_matrix(log_category='transformed-dataset-train-charts')

        # #
        # feature_array = train_transformed_dataset.get_feature_array(feature_name='monitoring-cycle')
        # train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='train-monitoring-cycle')
        # train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name='train-monitoring-cycle')
        # train_transformed_dataset_analysis.log_scatterchart(feature_name='monitoring-cycle', log_category='transformed-dataset-train-charts')
        # for ss in selected_settings:
        #         feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
        #         train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')
        # for ss in selected_sensors:
        #         feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
        #         train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')
        # for ss in train_transformed_dataset.selected_sensors_time_derivative:
        #         feature_array = train_transformed_dataset.get_feature_array(feature_name=ss)
        #         train_transformed_dataset_analysis.log_violinchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_boxchart(feature_array, log_category='transformed-dataset-train-charts', plot_name=f'train-{ss}')
        #         train_transformed_dataset_analysis.log_scatterchart(feature_name=ss, log_category='transformed-dataset-train-charts')



        # Modeling
        train_preprocess_pipeline = modeling.DataPreprocessPipeline(log=log, transformed_dataset=train_transformed_dataset)
        train_modeling_pipeline = modeling.ModelingPipeline(log=log, data_preprocess_pipeline=train_preprocess_pipeline)
        train_model = modeling.Model(log=log, modeling_pipeline=train_modeling_pipeline)











if __name__ == '__main__':

        #parser = argparse.ArgumentParser(description='Predictive maintenance prediction.')

        #parser.add_argument('--neptune-token', dest='neptune_token', type=str, nargs='?', help='Neptune token to access you account and run the experiment.')
        #parser.add_argument('--neptune-project-name', dest='neptune_project_name', type=str, nargs='?', help='Neptune qualified project name.')
        #parser.add_argument('--neptune-experiment-name', dest='neptune_experiment_name', type=str, nargs='?', help='Neptune experiment name.')

        #args = parser.parse_args()
        args = ns.Args()
        
        main(args)       