from operator import truediv
import neptune
from getpass import getpass 


class Args:

        def __init__(self):

                #
                self.neptune_token = getpass('Neptune token:')
                self.neptune_project_name = 'vitorvilela/suzano'
                self.experiment_name = 'dev'

                #
                self.origin_cycle_period = 1
                self.filter_window_size = 10
                self.monitoring_cycle_step = int(0.5*self.filter_window_size)
                self.n_monitoring_cycles_per_asset = 10

                #
                self.option_polynomial_features = True
                self.polynomial_features_degree = 2   
                self.option_standardize = True

                #
                self.model_scores = [('NEG-MAX', 'max_error'), ('NEG-MSE', 'neg_mean_squared_error'), ('R2', 'r2')]
        






class Neptune:

        def __init__(self, args):
                
                self.args = args

                self.PARAMS = {}

                self.create_experiment(args)


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

                self.exp = neptune.create_experiment(self.experiment_name, params=self.PARAMS, upload_source_files=['*.py'])
                self.exp.set_property('experiment-name', self.experiment_name)

                self.exp.log_artifact('./dataset')