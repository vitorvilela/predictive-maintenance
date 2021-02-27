import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm



class Analysis:

        def __init__(self, log):
                """
                Info
                """

                self.log = log
                self.args = log.args

        
        def log_violinchart(self, array, log_category, plot_name):
                """
                Info
                """   
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.violinplot(array, showmeans=False, showmedians=True)                
                ax.set_xlabel(plot_name)                
                self.log.exp.log_image(f'{log_category}', fig, image_name=f'{plot_name}-violinchart')
                plt.close(fig)


        def log_boxchart(self, array, log_category, plot_name):
                """
                Info
                """
                
                fig = plt.figure(figsize=(4., 3.), dpi=300)
                ax = fig.add_subplot(111)
                ax.boxplot(array)                
                ax.set_xlabel(plot_name)                
                self.log.exp.log_image(f'{log_category}', fig, image_name=f'{plot_name}-boxchart')
                plt.close(fig)   




class DatasetAnalysis(Analysis):

        def __init__(self, log, dataset):
                """
                Info
                """

                self.log = log
                self.args = log.args  
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
                self.log.exp.log_image(f'{self.dataset.type}-features-charts', fig, image_name=f'{self.dataset.type}-{feature_name}-for-asset-{asset_id}-linechart')
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
                self.log.exp.log_image(f'{self.dataset.type}-features-charts', fig, image_name=f'{self.dataset.type}-failure-values-of-{sensor_name}-for-assets-linechart')
                plt.close(fig)




class TransformedDatasetAnalysis(Analysis):

        def __init__(self, log, transformed_dataset):
                """
                Info
                """

                self.log = log
                self.args = log.args  
                self.transformed_dataset = transformed_dataset

                self.correlation_method = 'spearman'
                self.correlation_matrix_bar_scale = 1.0


        def log_correlation_matrix(self, log_category):

                if self.transformed_dataset.type=='train':
  
                        fig = plt.figure(figsize=(16., 9.), dpi=300)
                        ax = fig.add_subplot(111)
                        # Optins: 'Greys', 'jet'
                        cmap = cm.get_cmap('RdYlGn', lut=30) 
                        labels = self.transformed_dataset.dataset_header[1:] 
                        # Option: interpolation='nearest'
                        cax = ax.imshow(self.transformed_dataset.dataframe[labels].corr(method=self.correlation_method), vmin=-self.correlation_matrix_bar_scale, vmax=self.correlation_matrix_bar_scale, interpolation='None', cmap=cmap)
                        #ax.grid(True)
                        #plt.title('Title', fontsize=18)  
                        ticks = np.arange(0, len(labels), 1)
                        ax.set_xticks(ticks)
                        ax.set_yticks(ticks)
                        ax.set_xticklabels(labels, color='k', fontweight='normal', fontsize=8, fontstyle='italic', rotation='90')
                        ax.set_yticklabels(labels, color='k', fontweight='normal', fontsize=8, fontstyle='italic')
                        # Option: ticks=[-1, -0,75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                        cbar = fig.colorbar(cax, shrink=0.8, aspect=20, fraction=.15, pad=.03) 
                        cbar.set_label(self.correlation_method, size=10)
                        cbar.ax.tick_params(labelsize=8) 
                        plt.tight_layout()
                        self.log.exp.log_image(log_category, fig, image_name=f'{self.transformed_dataset.type}-correlationmatrix')
                        plt.close(fig)
                else:
                        print('on correlation_matrix')      
                        sys.exit(1) 


        def log_scatterchart(self, feature_name, log_category):

                df = self.transformed_dataset.dataframe

                fig = plt.figure(figsize=(4., 3.), dpi=1200)
                ax = fig.add_subplot(111)

                for x, y, i in zip(df.loc[:, feature_name], df.loc[:, 'rul'], df['monitoring-cycle']):              
                        ax.plot(x, y, 'o', color='g', alpha=0.2, markersize=2)
                        ax.annotate(i, (x, y), size=1, color='k')            
                   
                plt.xlabel(feature_name, color='k', fontweight='normal', fontsize=8, fontstyle='italic')
                plt.ylabel('rul', color='k', fontweight='normal', fontsize=8, fontstyle='italic', rotation='vertical')  
                ax.tick_params(labelsize=6) 
                plt.tight_layout()
                self.log.exp.log_image(log_category, fig, image_name=f'{self.transformed_dataset.type}-{feature_name}-scatterchart')
                plt.close(fig)