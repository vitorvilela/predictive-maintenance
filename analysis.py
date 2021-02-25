import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.api.table import log_table


class Analysis:

        def __init__(self, dataset):
                """
                Info
                """

                self.dataset = dataset
                self.n_assets = self.get_assets_quantity()
                                

        def get_assets_quantity(self):
                """
                How many assets there are in the dataset?
                Returns: an integer
                """
                df = self.dataset.dataframe
                n_assets = len(self.dataset.assets)
                return n_assets

        def get_assets_last_cycle_array(self):
                """
                Info
                """

                df = self.dataset.dataframe              
                assets_last_cycle_array = np.array([self.dataset.get_asset_last_cycle(a) for a in self.dataset.assets])

                return assets_last_cycle_array

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


        def get_dummy_mean_precision(self, type='mean'):
                """
                Info
                """

                assets_last_cycle_array = self.get_assets_last_cycle_array()
                
                if 'assets_last_cycle_dict' not in locals(): 
                        self.compute_assets_last_cycle_statistics()

                dummy_mean_precision = 0.

                if type=='mean':
                        dummy_precision_array = 1. - np.abs(assets_last_cycle_array -  self.assets_last_cycle_dict['mean']) / self.assets_last_cycle_dict['mean']
                        dummy_mean_precision = np.mean(dummy_precision_array)
                elif type=='min':
                        dummy_precision_array = 1. - np.abs(assets_last_cycle_array -  self.assets_last_cycle_dict['min']) / self.assets_last_cycle_dict['min']
                        dummy_mean_precision = np.mean(dummy_precision_array)                
                else:
                        print(f'In get_dummy_precision(), there is not the input ({type}) available.')
                        sys.exit(1)  

                # TODO Log dummy_mean_precision        

                return dummy_mean_precision        


        def log_feature_linechart_for_asset(self, asset_id, feature_name):
                """
                Info
                """               
                
                try:
                        arrays = self.dataset.get_cycle_feature_array(asset_id, feature_name)
                except:
                        print(f'In log_feature_linechart_for_asset(), there is not the input ({asset_id}, {feature_name}) in the dataset.')
                        sys.exit(1)  

                cycle_array = arrays[:,0]
                feature_array = arrays[:,1]           

                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.plot(cycle_array, feature_array)                
                plt.xlabel('cycle')
                plt.ylabel(f'{feature_name}')
                plt.tight_layout()
                neptune.log_image(f'features-charts', fig, image_name=f'{feature_name}-for-asset-linechart{asset_id}')
                plt.close(fig)
                

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


        def get_sensors_last_value_for_assets(self, sensor_name):
                """
                Info
                """

                sensor_last_values_list = []

                for asset in self.dataset.assets:

                        df = self.dataset.get_asset_dataframe(asset_id=asset)
                        
                        try:        
                                sensor_array = df[sensor_name].values
                        except:
                                print(f'In get_sensors_last_value_for_assets(), there is not the input tuple ({asset}, {sensor_name}) in the dataset.')
                                sys.exit(1) 

                        sensor_last_values_list.append([asset, sensor_array[-1]])

                return np.array(sensor_last_values_list)  


        def log_sensor_failure_value_linechart_for_assets(self, sensor_name):
                """
                Info
                """

                sensor_failure_values_array = self.get_sensors_last_value_for_assets(sensor_name)
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.plot(sensor_failure_values_array[:, 0], sensor_failure_values_array[:, 1])                
                plt.xlabel('asset')
                plt.ylabel(f'{sensor_name}')
                plt.tight_layout()
                neptune.log_image(f'features-charts', fig, image_name=f'failure-values-of-{sensor_name}-for-assets-linechart')
                plt.close(fig)


