import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm, trange

from models import *
from data_utils import data_to_gpu

import pdb

class Trainer(object):

    def __init__(self, args, combat_list):

        self.args = args
        self.combat_list = combat_list
        
        xs, y_int, y_atk, squad2squad_idx, entity2squad_idx, tactic_id, combat_id = combat_list

        Tc = 0 # combat dimension
        i = 1 # entity dimension
        t = 2 # timestep dimension
        d0 = 3 # feature dimension

        self.num_Tc = xs.shape[Tc] # num_combat
        self.num_i = xs.shape[i] # num_entity
        self.num_t = xs.shape[t] # num_timestep
        self.num_d0 = xs.shape[d0] # num_feature

        self.num_tactics = int(tactic_id.max() + 1) # num_tactics
        self.num_Sj = entity2squad_idx.shape[-1] # num_squads
        self.num_int_labels = y_int.shape[-1] # num_intention_labels


    def init_model(self):

        if self.args.model == 'SAFETY' : model = Spatio_Temporal_Att(self.args, self.num_d0, self.num_int_labels).to(self.args.device)
        if self.args.model == 'GRU': model = GRU_Classifier(self.args, self.num_d0, self.num_int_labels).to(self.args.device)
        if self.args.model == 'LSTM': model = LSTM_Classifier(self.args, self.num_d0, self.num_int_labels).to(self.args.device)
        if self.args.model == 'P-LSTM': model = Panoramic_LSTM(self.args, self.num_d0, self.num_int_labels).to(self.args.device)
        if self.args.model == 'MLP' : model = MLP_Classifier(self.args, self.num_d0, self.num_int_labels).to(self.args.device)

        return model

    def train(self, train_set, model, y_int_weight, y_atk_weight, squad2squad_idx):

        iterator = trange(self.args.num_epochs, desc='Train Loss: ', leave=False)
        best_loss = np.inf
        optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr, weight_decay = self.args.dr)

        iter_counter = 0
        for epoch in iterator:
            epoch_loss = []
            model.train()
            
            for train_batch in train_set:
                features, int_label, atk_label, squad2squad_idx, entity2squad_idx, _, _, y_int_weight, y_atk_weight = data_to_gpu(self.args.device, train_batch, y_int_weight, y_atk_weight, squad2squad_idx)
            
                int_prob, atk_prob = model(features, entity2squad_idx, squad2squad_idx)

                train_loss = self.loss_cal(int_prob, int_label, atk_prob, atk_label, y_int_weight, y_atk_weight)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss.append(train_loss.item())
                iter_counter += 1

            epoch_loss = np.mean(epoch_loss)
            iterator.set_description("Loss: {:.4f}".format(epoch_loss))

            # early stopping
            step_counter = 0
            if epoch_loss < best_loss:
                step_counter = 0
                best_loss = epoch_loss
                best_model = model
            else:
                step_counter += 1
                if step_counter > self.args.patience: break
            
        return best_model

    def loss_cal(self, int_prob, int_label, atk_prob, atk_label, y_int_weight, y_atk_weight):  

        int_criterion = nn.CrossEntropyLoss(weight = y_int_weight)
        atk_criterion = nn.BCELoss(weight = y_atk_weight)

        if self.args.pred == 'y_int':
            loss = int_criterion(int_prob.flatten(0,1), int_label.flatten(0,1))

        elif self.args.pred == 'y_atk':
            loss = atk_criterion(atk_prob, atk_label)

        return loss

    def fit_model(self, train_set, y_int_weight, y_atk_weight, squad2squad_idx):
        
        model = self.init_model()
        best_model = self.train(train_set, model, y_int_weight, y_atk_weight, squad2squad_idx)

        return best_model

