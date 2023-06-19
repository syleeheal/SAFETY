import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import pickle
import random

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import os
import gdown


def label_weights_generation(train_data):

    xs, y_task, y_attack, edge_index, node2unit_mat, scenario_id, simulation_id = train_data
    
    num_graphs = y_attack.shape[0]
    num_units = y_attack.shape[1]
    
    task_label_num = torch.sum(y_task, dim = (0,1))
    attack_label_num = torch.sum(y_attack)
    
    task_label_weight = 1/task_label_num
    task_label_weight[torch.isinf(task_label_weight)] = 0
    task_label_weight = task_label_weight / torch.sum(task_label_weight)
    
    attack_label_weight = attack_label_num / (num_graphs * num_units * num_units)

    return task_label_weight, attack_label_weight

def load_graph_list(args):

    scenario_id_list = []
    simulation_id_list = []
    
    xs_list = []
    y_task_list = []
    y_attack_list = []
    node2unit_mat_list = []

    url = 'https://drive.google.com/uc?export=download&id=13TMlyQCuQDzWWsC2e8lAw96g4ndOdrjx'
    output = './SAFETY/dataset/data_dict.pkl'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, proxy=None)
    

    with open('./SAFETY/dataset/data_dict.pkl', 'rb') as f: data_dict = pickle.load(f)


    # find max length of time steps
    max_time_steps = 700

    num_scenarios = len(data_dict)
    for scenario_id in range(num_scenarios):

        num_simulations = len(data_dict[scenario_id])
        for simulation_id in range(num_simulations):

            xs_node = []

            num_nodes = len(data_dict[scenario_id][simulation_id]['node_id'])
            for node in range(num_nodes):

                x_node = data_dict[scenario_id][simulation_id]['node_id'][node]['x']
                xs_node.append(x_node)

            xs_node = np.array(xs_node)
            xs_node = torch.FloatTensor(xs_node)
            # pad if shorter than max_time_steps
            if xs_node.shape[1] < max_time_steps:
                xs_node = F.pad(xs_node, (0,0,0,max_time_steps-xs_node.shape[1],0,0), 'constant', 0)


                y_task = []
            y_attack = []
            
            num_units = len(data_dict[scenario_id][simulation_id]['unit_id'])
            for unit in range(num_units):

                y_task_unit = data_dict[scenario_id][simulation_id]['unit_id'][unit]['y_task']
                y_task.append(y_task_unit) 

                y_attack_unit = data_dict[scenario_id][simulation_id]['unit_id'][unit]['y_attack']
                y_attack.append(y_attack_unit)
            
            y_task = np.array(y_task)
            y_task = torch.FloatTensor(y_task)

            y_attack = np.array(y_attack)
            y_attack = torch.FloatTensor(y_attack)



            node2unit_idx = []

            num_nodes = len(data_dict[scenario_id][simulation_id]['node_id'])
            for node in range(num_nodes): 
                node2unit_idx.append([node, data_dict[scenario_id][simulation_id]['node_id'][node]['unit']])
            
            node2unit_idx = np.array(node2unit_idx)
            node2unit_idx = torch.LongTensor(node2unit_idx).T
            node2unit_mat = F.one_hot(node2unit_idx[1]).float()
            
            

            xs_list.append(xs_node.unsqueeze(0))
            y_task_list.append(y_task.unsqueeze(0))
            y_attack_list.append(y_attack.unsqueeze(0))
            node2unit_mat_list.append(node2unit_mat.unsqueeze(0))
            
            scenario_id_list.append(scenario_id)
            simulation_id_list.append(simulation_id)

    
    
    scenario_id = torch.LongTensor(scenario_id_list)
    simulation_id = torch.LongTensor(simulation_id_list)

    xs = torch.cat(xs_list, dim = 0)
    y_task = torch.cat(y_task_list, dim = 0)
    y_attack = torch.cat(y_attack_list, dim = 0)
    node2unit_mat = torch.cat(node2unit_mat_list, dim = 0)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes)], dtype = torch.long).t().contiguous()


    graph_list = [xs, y_task, y_attack, edge_index, node2unit_mat, scenario_id, simulation_id]

    return graph_list

def feature_preprocessing(args, x, mean, std):

    xs_node = x.clone()

    if args.time_init == 'fixed': 
        xs_node = xs_node[:,:, :args.time_steps, :]
    elif args.time_init == 'random': 
        non_zero_idx = torch.nonzero(torch.sum(xs_node, dim = (0,2))).squeeze()
        end_time_step = random.randint(args.time_steps, non_zero_idx.shape[0])
        xs_node = xs_node[:,:, end_time_step-args.time_steps : end_time_step, :]
    elif args.time_init == 'intermittent': 
        non_zero_idx = torch.nonzero(torch.sum(xs_node, dim = (0,2))).squeeze()
        end_time_step = random.randint(args.time_steps, non_zero_idx.shape[0])
        
        time_indices = random.sample(range(end_time_step), args.time_steps)
        time_indices.sort()
        xs_node = xs_node[:,:, time_indices, :]


    # x[:,:,0:3] = 'PositionLat(deg)', 'PositionLon(deg)', 'PositionAlt(m)'
    # x[:,:,3:6] ='AttitudeYaw(deg)', 'AttitudePitch(deg)', 'AttitudeRoll(deg)'
    # x[:,:,6] = Speed (km/\h)
    # x[:,:,7] = Force Identifier
    # x[:,:,8:] = 'T_Road', 'T_Forest', 'T_OpenLane', 'T_HidingPlace', 'T_Building'

    xs_node[:,:,:,7] = xs_node[:,:,:,7] - 1
    xs_node[:,:,:,:7] = (xs_node[:,:,:,:7] - mean) / std

    xs_node = torch.where(torch.isnan(xs_node), torch.zeros_like(xs_node), xs_node)

    return xs_node
