import optuna
import torch
import numpy as np
import random
import pdb
from train import Trainer
from eval import infer, eval_atk, eval_int, save_results
from data_utils import data_loader

class Hyperparam_Optimizer(object):

    def __init__(self, args, combat_list):

        self.args = args
        self.combat_list = combat_list
        
    def objective(self, trial):
        """
        Objective function for hyperparameter optimization.
        """

        self.args.dr = trial.suggest_categorical('dr', [5e-4, 1e-4, 5e-4, 1e-5, 5e-6, 1e-6])
        self.args.dropout = trial.suggest_categorical('dropout', [0.3, 0.5, 0.7])
        if self.args.model != 'SAFETY':
            self.args.num_layers = trial.suggest_categorical('num_layers', [2, 4])

        
        """
        experiment
        """
        auroc_scores = []
        f1_micro_scores, f1_macro_scores = [], []

        seeds = torch.load('./SAFETY/seeds_100.pt')
        for exp in range(3):

            # fix random seed
            seed = seeds[exp]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # load data
            train_set, test_set, y_int_weight, y_atk_weight, squad2squad_index = data_loader(self.args, int(exp*10), self.combat_list)

            # train model
            best_model = Trainer(self.args, self.combat_list).fit_model(train_set, y_int_weight, y_atk_weight, squad2squad_index)

            # evaluate and save results
            y_int_prob, y_atk_prob, y_int_label, y_atk_label = infer(self.args, test_set, best_model, squad2squad_index)

            if self.args.pred in 'y_atk': 
                auroc = eval_atk(exp, y_atk_prob, y_atk_label)
                auroc_scores.append(auroc)

            if self.args.pred == 'y_int': 
                f1_macro, f1_micro = eval_int(exp, y_int_prob, y_int_label)
                f1_micro_scores.append(f1_micro)
                f1_macro_scores.append(f1_macro)
        
        mean_perf = np.mean(auroc_scores) if self.args.pred == 'y_atk' else np.mean(f1_macro_scores)
        
        return mean_perf


    def search(self):
        if self.args.model != 'SAFETY':
            search_space = {
                            'dr': [5e-4, 1e-4, 5e-4, 1e-5, 5e-6, 1e-6],
                            'dropout': [0.3, 0.5, 0.7],
                            'num_layers': [2, 4],
                            }

        else:
            search_space = {
                            'dr': [5e-4, 1e-4, 5e-4, 1e-5, 5e-6, 1e-6],
                            'dropout': [0.3, 0.5, 0.7],
                            }


        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(self.objective, n_trials=300,)

        print('\n \n')

        print('model: {}'.format(self.args.model))
        print("\nBest hyperparameters: {}".format(study.best_params))
        print("\nBest test accuracy: {:4f}".format(study.best_value))
        print('\n Parameter importance: \n')
        print(optuna.importance.get_param_importances(study))
        

        # write results to file
        file = open('./SAFETY/best_hyperparam/' + self.args.model + '_' + self.args.pred + '.txt', 'w')
        file.write('Model: {}\n'.format(self.args.model))
        file.write('Best hyperparameters: {}\n'.format(study.best_params))
        file.write('Best val accuracy: {:4f}\n'.format(study.best_value))
        file.write('Parameter importance: {}\n'.format(optuna.importance.get_param_importances(study)))
        file.write('\n')
        

        print('\n \n')

        print('Best trials:')
        for trial in study.best_trials[:5]: 
            print('  Value: {}'.format(trial.value))
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

