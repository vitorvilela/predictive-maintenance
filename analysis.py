import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neptune



class Analysis:

        def __init__(self):
                pass


        def log_violinchart(self, array, log_category, plot_name):
                """
                Info
                """   
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.violinplot(array, showmeans=False, showmedians=True)                
                ax.set_xlabel(plot_name)                
                neptune.log_image(f'{log_category}', fig, image_name=f'{plot_name}-violinchart')
                plt.close(fig)


        def log_boxchart(self, array, log_category, plot_name):
                """
                Info
                """
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.boxplot(array)                
                ax.set_xlabel(plot_name)                
                neptune.log_image(f'{log_category}', fig, image_name=f'{plot_name}-boxchart')
                plt.close(fig)   


                



class DatasetAnalysis(Analysis):

        def __init__(self, dataset):
                """
                Info
                """

                self.dataset = dataset
               

        def get_dummy_mean_precision(self, type='mean'):
                """
                Info
                """

                assets_last_cycle_array = self.dataset.get_assets_last_cycle_array()
                
                dummy_mean_precision = 0.

                if type=='mean':
                        dummy_precision_array = 1. - np.abs(assets_last_cycle_array -  self.dataset.assets_last_cycle_dict['mean']) / self.dataset.assets_last_cycle_dict['mean']
                        dummy_mean_precision = np.mean(dummy_precision_array)
                elif type=='min':
                        dummy_precision_array = 1. - np.abs(assets_last_cycle_array -  self.dataset.assets_last_cycle_dict['min']) / self.dataset.assets_last_cycle_dict['min']
                        dummy_mean_precision = np.mean(dummy_precision_array)                
                else:
                        print(f'In get_dummy_precision(), the type={type} is not available. Please use \'mean\' or \'min\'.')
                        sys.exit(1)  

                return dummy_mean_precision        


        def log_feature_linechart_for_asset(self, asset_id, feature_name):
                """
                Info
                """               
                
                arrays = self.dataset.get_cycle_feature_array_for_asset(asset_id, feature_name)
                cycle_array = arrays[:,0]
                feature_array = arrays[:,1]           

                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.plot(cycle_array, feature_array)                
                plt.xlabel('cycle')
                plt.ylabel(f'{feature_name}')
                plt.tight_layout()
                neptune.log_image(f'{self.dataset.type}-features-charts', fig, image_name=f'{self.dataset.type}-{feature_name}-for-asset-linechart{asset_id}')
                plt.close(fig)
                

        def log_sensor_failure_value_linechart_for_assets(self, sensor_name):
                """
                Info
                """

                sensor_failure_values_array = self.dataset.get_sensors_last_value_for_assets(sensor_name)
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.plot(sensor_failure_values_array[:, 0], sensor_failure_values_array[:, 1])                
                plt.xlabel('asset')
                plt.ylabel(f'{sensor_name}')
                plt.tight_layout()
                neptune.log_image(f'{self.dataset.type}-features-charts', fig, image_name=f'{self.dataset.type}-failure-values-of-{sensor_name}-for-assets-linechart')
                plt.close(fig)