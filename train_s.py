import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score

import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
import xgboost as xgb

from models import *
from data_utils import label_weights_generation, load_graph_list, feature_preprocessing
from utils import masking, add_noise, node2unit_pool

import pdb


class Trainer_S(object):

    def __init__(self, args, graph_list):
        self.args = args
        self.graph_list = graph_list
        torch.cuda.set_device(self.args.device)

        xs, y_task, y_attack, edge_index, node2unit_mat, scenario_id, simulation_id = graph_list

        self.args.num_unit_labels = y_task.shape[-1]
        self.args.num_node_feat = xs.shape[-1] 
        self.args.num_nodes = xs.shape[0] 
        self.args.num_units = node2unit_mat.shape[-1]
        self.args.num_graphs = len(simulation_id) 
        self.args.num_scenario = int(scenario_id.max() + 1)
        
    def init_model(self, train_set):
        if self.args.model == 'kNN': 
            model_task = NearestCentroid()
            model_attack = NearestCentroid()
        elif self.args.model == 'XG':
            self.class_weights, ones_proportion = label_weights_generation(train_set)
            model_task = xgb.XGBClassifier(objective='multi:softmax', n_estimators=self.args.num_estimators, random_state=self.seed, num_class=self.args.num_unit_labels)
            model_attack = xgb.XGBClassifier(objective='binary:logistic', n_estimators=self.args.num_estimators, random_state=self.seed, scale_pos_weight=ones_proportion.numpy())
        return model_task, model_attack

    def feat_transform(self, x, node2unit_mat):
        x = feature_preprocessing(self.args, x, self.coord_mean, self.coord_std)
        x = x.permute(0, 2, 1, 3)

        if self.args.stress_type == 'noise':
            x = add_noise(x, self.args.noise)
        elif self.args.stress_type == 'mask':
            x = masking(x, node2unit_mat, self.args.mask_prob, self.args.mask_type)
        elif self.args.stress_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x

    def train_task(self, train_set, val_set, test_set, model,):
        
        features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = train_set
        
        # concatenate time dim
        features = features.reshape(features.shape[0], features.shape[2], -1)
        # unit aggregation
        features = node2unit_pool(features, node_to_unit)

        # concatenate unit dim
        features = features.reshape(-1, features.shape[-1])
        behavior_label = behavior_label.reshape(-1, behavior_label.shape[2])
        behavior_label = np.argmax(behavior_label, axis=1)

        if self.args.model == 'XG':
            sample_weight = np.array([self.class_weights[i] for i in behavior_label])
            model.fit(features, behavior_label, sample_weight=sample_weight)
        else:
            model.fit(features, behavior_label)

        return model

    def train_attack(self, train_set, val_set, test_set, model,):
        
        features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = train_set

        # concatenate time dim
        features = features.reshape(features.shape[0], features.shape[2], -1)        
        # unit aggregation
        features = node2unit_pool(features, node_to_unit)
        
        # aggregate for each unit pair
        unit_adj_matrix = torch.ones(self.args.num_units, self.args.num_units)
        unit_adj_idx = unit_adj_matrix.nonzero().t()
        features = features[:, unit_adj_idx[0], :] * features[:, unit_adj_idx[1], :]
        
        # concatenate unit dim
        features = features.reshape(-1, features.shape[-1])
        attack_label = attack_label.reshape(-1,)

        model.fit(features, attack_label)

        return model

    def eval(self, test_set, model_task, model_attack):

        features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = test_set
        behavior_label = behavior_label.numpy()
        attack_label = attack_label.numpy()

        behavior_label = behavior_label.reshape(-1, behavior_label.shape[2])
        behavior_label = np.argmax(behavior_label, axis=1)
        attack_label = attack_label.reshape(-1,)


        # concatenate time dim
        features = features.reshape(features.shape[0], features.shape[2], -1)
        # unit aggregation
        features = node2unit_pool(features, node_to_unit)
        # concatenate unit dim to get final unit features
        x = features.reshape(-1, features.shape[-1])
        

        # aggregate for each unit pair
        unit_adj_matrix = torch.ones(self.args.num_units, self.args.num_units)
        unit_adj_idx = unit_adj_matrix.nonzero().t()
        edge_x = features[:, unit_adj_idx[0], :] * features[:, unit_adj_idx[1], :]
        # concatenate unit dim to get final unit pair features
        edge_x = edge_x.reshape(-1, edge_x.shape[-1])


        # predict
        behavior_prob = model_task.predict(x)
        attack_prob = model_attack.predict(edge_x)


        # eval
        f1_macro = f1_score(behavior_label, behavior_prob, average='macro')
        f1_micro = f1_score(behavior_label, behavior_prob, average='micro')
        acc = (behavior_label == behavior_prob).sum().item() / len(behavior_label)
        auroc = roc_auc_score(attack_label, attack_prob)

        self.auroc= auroc
        self.f1_micro = f1_micro
        self.f1_macro = f1_macro

        return auroc, f1_macro, f1_micro

    def data_loader(self, graph_list):
        xs, y_task, y_attack, edge_index, node2unit_mat, scenario_id, simulation_id = graph_list

        train_idx = list(range(self.args.num_scenarios))
        test_idx = train_idx[0:3]
        random.shuffle(train_idx)
        random.shuffle(test_idx)
        train_idx.remove(test_idx[0])

        test_idx = (scenario_id == test_idx[0]).nonzero().view(-1)
        
        train_idx1 = (scenario_id == train_idx[0]).nonzero().view(-1)
        train_idx2 = (scenario_id == train_idx[1]).nonzero().view(-1)
        train_idx3 = (scenario_id == train_idx[2]).nonzero().view(-1)

        train_idx = torch.cat((train_idx1, train_idx2, train_idx3), dim=0)

        self.coord_mean = torch.mean(xs[train_idx][:,:,:,:7], dim = (0,1,2))
        self.coord_std = torch.std(xs[train_idx][:,:,:,:7], dim = (0,1,2))

        train_xs = self.feat_transform(xs[train_idx], node2unit_mat[train_idx])
        test_xs = self.feat_transform(xs[test_idx], node2unit_mat[test_idx])

        train_set = [train_xs, y_task[train_idx], y_attack[train_idx], edge_index, node2unit_mat[train_idx], scenario_id[train_idx], simulation_id[train_idx]]
        test_set = [test_xs, y_task[test_idx], y_attack[test_idx], edge_index, node2unit_mat[test_idx], scenario_id[test_idx], simulation_id[test_idx]]
        val_set = None

        return train_set, val_set, test_set

    def fit_model(self):

        seeds = torch.load('./seeds_100.pt')
        
        for exp in tqdm(range(self.args.num_exp)):
            
            self.seed = seeds[exp]
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            sklearn.utils.check_random_state(self.seed)


            train_set, val_set, test_set = self.data_loader(self.graph_list)
            
            model_task, model_attack = self.init_model(train_set)
            best_model_task = self.train_task(train_set, val_set, test_set, model_task)
            best_model_attack = self.train_attack(train_set, val_set, test_set, model_attack)

            auroc, f1_macro, f1_micro = self.eval(test_set, best_model_task, best_model_attack)
            self.save_results(auroc, f1_macro, f1_micro, exp)

        self.print_results()

    def save_results(self, auroc, f1_macro, f1_micro, exp):

        if exp == 0:
            self.f1_micro_scores = []
            self.f1_macro_scores = []
            self.auroc_scores = []

        self.auroc_scores.append(auroc)
        self.f1_micro_scores.append(f1_micro)
        self.f1_macro_scores.append(f1_macro)
        self.mean_performance = (np.mean(self.auroc_scores) + np.mean(self.f1_macro_scores)) / 2

    def print_results(self):

        print('Model: {}'.format(self.args.model))
        print('Predict: {}'.format(self.args.pred))
        print('Time Steps: {}'.format(self.args.time_steps))
        print('Time Init: {}'.format(self.args.time_init))
        print('Stress Type: {}'.format(self.args.stress_type))
        if self.args.stress_type == 'noise': 
            print('Noise Level: {}'.format(self.args.noise))
        elif self.args.stress_type == 'mask': 
            print('Mask Type: {}'.format(self.args.mask_type))
            print('Mask Prob: {}'.format(self.args.mask_prob))
        print()

        if self.args.pred in ['attack', 'joint']:
            print("Mean AUROC: {:.2f} ± {:.2f}".format(np.mean(self.auroc_scores)*100, np.std(self.auroc_scores)*100))
        if self.args.pred in ['task', 'joint']:
            print("Mean F1-Micro: {:.2f} ± {:.2f}".format(np.mean(self.f1_micro_scores)*100, np.std(self.f1_micro_scores)*100))
            print("Mean F1-Macro: {:.2f} ± {:.2f}".format(np.mean(self.f1_macro_scores)*100, np.std(self.f1_macro_scores)*100))
        print()

