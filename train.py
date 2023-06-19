import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score

from models import *
from data_utils import label_weights_generation, load_graph_list, feature_preprocessing
from utils import masking, add_noise

import pdb


class Trainer(object):

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
         
    def init_model(self):

        if self.args.model == 'STA' : model = Spatio_Temporal_Att(self.args).to(self.args.device)
        if self.args.model == 'GRU': model = GRU_Classifier(self.args).to(self.args.device)
        if self.args.model == 'RNN' : model = RNN_Classifier(self.args).to(self.args.device)
        if self.args.model == 'P-LSTM': model = Panoramic_LSTM(self.args).to(self.args.device)
        if self.args.model == 'MLP' : model = MLP_Classifier(self.args).to(self.args.device)

        return model

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

    def train(self, train_set, val_set, test_set, model,):

        iterator = trange(self.args.num_epochs, desc='Train Loss: ', leave=False)
        best_loss = np.inf
        optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr, weight_decay = self.args.dr)

        step_counter = 0
        class_weights, ones_proportion = label_weights_generation(train_set)

        for epoch in iterator:
            model.train()
                            
            features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = self.to_gpu(train_set)
           
            behavior_prob, attack_prob = model(features, edge_index, node_to_unit)
            train_loss = self.loss_cal(behavior_prob, behavior_label, attack_prob, attack_label, class_weights, ones_proportion)                 
            train_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
                        
            if val_set is not None:
                model.eval()
                with torch.no_grad():

                    features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = self.to_gpu(val_set)

                    behavior_prob, attack_prob = model(features, edge_index, node_to_unit)
                    val_loss = self.loss_cal(behavior_prob, behavior_label, attack_prob, attack_label, class_weights, ones_proportion) 
            
            else:
                val_loss = train_loss

            iterator.set_description("Loss: {:.4f}".format(val_loss))

            # early stopping
            if val_loss < best_loss:
                step_counter = 0
                best_loss = val_loss
                best_model = model

            else:
                step_counter += 1
                if step_counter >= self.args.patience:
                    break
            
        return best_model

    def eval(self, test_set, model):

        features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = self.to_gpu(test_set)

        model.eval()
        with torch.no_grad():
            
            behavior_prob, attack_prob = model(features, edge_index, node_to_unit)
            if self.args.pred == 'attack':
                auroc = roc_auc_score(attack_label.cpu().numpy().reshape(-1), attack_prob.cpu().numpy().reshape(-1))
                print("Attack Prediction Performance : auroc: {:.4f}".format(auroc))

                f1_macro = None
                f1_micro = None

            elif self.args.pred == 'task':
                behavior_pred = torch.max(behavior_prob, dim=-1)[1].cpu().numpy().reshape(-1)
                behavior_label = torch.max(behavior_label, dim=-1)[1].cpu().numpy().reshape(-1)

                f1_macro = f1_score(behavior_label, behavior_pred, average='macro')
                f1_micro = f1_score(behavior_label, behavior_pred, average='micro')
                print("Task Prediction : f1_macro: {:.4f}, f1_micro: {:.4f}".format(f1_macro, f1_micro))
                auroc = None

            elif self.args.pred == 'joint':
                behavior_pred = torch.max(behavior_prob, dim=-1)[1].cpu().numpy().reshape(-1)
                behavior_label = torch.max(behavior_label, dim=-1)[1].cpu().numpy().reshape(-1)

                auroc = roc_auc_score(attack_label.cpu().numpy().reshape(-1), attack_prob.cpu().numpy().reshape(-1))
                f1_macro = f1_score(behavior_label, behavior_pred, average='macro')
                f1_micro = f1_score(behavior_label, behavior_pred, average='micro')
                print("Attack Prediction Performance : auroc: {:.4f}".format(auroc))
                print("Task Prediction : f1_macro: {:.4f}, f1_micro: {:.4f}".format(f1_macro, f1_micro))

        return auroc, f1_macro, f1_micro

    def to_gpu(self, train_set):
        features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id = train_set
        features = features.to(self.args.device)
        behavior_label = behavior_label.to(self.args.device)
        attack_label = attack_label.to(self.args.device)
        node_to_unit = node_to_unit.to(self.args.device)
        edge_index = edge_index.to(self.args.device)
        
        return features, behavior_label, attack_label, edge_index, node_to_unit, scenario_id, simulation_id

    def loss_cal(self, behavior_prob, behavior_label, attack_prob, attack_label, class_weights, ones_proportion):  

        # behavior_prob shape = (num_samples, num_units, num_unit_labels)
        # behavior_label shape = (num_samples, num_units, num_unit_labels)
        # attack_prob shape = (num_samples, num_units, num_units)
        # attack_label shape = (num_samples, num_units, num_units)

        if self.args.pred == 'task':
            behavior_criterion = nn.CrossEntropyLoss(weight = class_weights.to(self.args.device))
            loss = behavior_criterion(behavior_prob.view(-1, self.args.num_unit_labels), behavior_label.view(-1, self.args.num_unit_labels))

        elif self.args.pred == 'attack':
            attack_criterion = nn.BCELoss(weight = (1-ones_proportion).to(self.args.device))
            loss = attack_criterion(attack_prob, attack_label)

        elif self.args.pred == 'joint':
            behavior_criterion = nn.CrossEntropyLoss(weight = class_weights.to(self.args.device))
            behavior_loss = behavior_criterion(behavior_prob.view(-1, self.args.num_unit_labels), behavior_label.view(-1, self.args.num_unit_labels))

            attack_criterion = nn.BCELoss(weight = (1-ones_proportion).to(self.args.device))
            attack_loss = attack_criterion(attack_prob, attack_label)

            loss = behavior_loss + attack_loss

        return loss

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
        
        for exp in range(self.args.num_exp):
            
            self.seed = seeds[exp]
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            train_set, val_set, test_set = self.data_loader(self.graph_list)
            
            model = self.init_model()
            best_model = self.train(train_set, val_set, test_set, model)
            auroc, f1_macro, f1_micro = self.eval(test_set, best_model)

            self.save_results(auroc, f1_macro, f1_micro, exp)

        self.print_results()

    def save_results(self, auroc, f1_macro, f1_micro, exp):

        if exp == 0:
            self.f1_micro_scores = []
            self.f1_macro_scores = []
            self.auroc_scores = []

        if self.args.pred in ['attack']: 
            self.auroc_scores.append(auroc)
            self.mean_performance = np.mean(self.auroc_scores)

        if self.args.pred in ['task']: 
            self.f1_micro_scores.append(f1_micro)
            self.f1_macro_scores.append(f1_macro)
            self.mean_performance = (np.mean(self.f1_micro_scores) + np.mean(self.f1_macro_scores)) / 2

        if self.args.pred in ['joint']:
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

