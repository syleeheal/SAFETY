import os
import pickle
import random
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pdb



def label_weights_generation(y_int, y_atk):
    """
    calculate label weights
    """
    
    num_Tc = y_atk.shape[0]
    num_Sj = y_atk.shape[1]
    
    int_label_num = torch.sum(y_int, dim = (0,1))
    atk_label_num = torch.sum(y_atk)
    
    int_label_weight = 1/int_label_num
    int_label_weight[torch.isinf(int_label_weight)] = 0
    int_label_weight = int_label_weight / torch.sum(int_label_weight)
    
    atk_label_weight = atk_label_num / (num_Tc * num_Sj * num_Sj)

    return int_label_weight, atk_label_weight

def load_combat_list(args):


    file_name = './dataset/data_dict.pkl'
    if not os.path.exists(file_name):
        with zipfile.ZipFile('./dataset/data_dict.zip', 'r') as zip_ref:
            zip_ref.extractall('./dataset/')
            print('Extracting data_dict.zip ...')
    
    with open('./dataset/data_dict.pkl', 'rb') as f: data_dict = pickle.load(f)


    """
    basic statistics
    """
    num_int = data_dict[0][0]['squad_id'][0]['y_int'].shape[-1]
    num_feat = data_dict[0][0]['entity_id'][0]['x'].shape[1]

    max_num_entities = 0
    max_num_squads = 0
    max_time_steps = 0

    for tactic_id in range(len(data_dict)):
        for combat_id in range(len(data_dict[tactic_id])):
            num_entities = len(data_dict[tactic_id][combat_id]['entity_id'])
            if num_entities > max_num_entities: max_num_entities = num_entities

            num_squads = len(data_dict[tactic_id][combat_id]['squad_id'])
            if num_squads > max_num_squads: max_num_squads = num_squads

            for entity in range(num_entities):
                x_entity = data_dict[tactic_id][combat_id]['entity_id'][entity]['x']
                if x_entity.shape[0] > max_time_steps: 
                    max_time_steps = x_entity.shape[0]

    """
    load combat list
    """
    combat_id_list, tactic_id_list = [], []
    xs_list, y_int_list, y_atk_list, entity2squad_idx_list = [], [], [], []

    num_tactic = len(data_dict)
    for tactic_id in range(num_tactic):

        num_combats = len(data_dict[tactic_id])
        for combat_id in range(num_combats):
            
            # get entity features
            xs_entity = []

            num_entities = len(data_dict[tactic_id][combat_id]['entity_id'])
            for entity in range(num_entities):
                x_entity = data_dict[tactic_id][combat_id]['entity_id'][entity]['x']
                # pad if shorter than max_time_steps
                if x_entity.shape[0] < args.t_max: x_entity = np.pad(x_entity, ((0, args.t_max - x_entity.shape[0]), (0,0)), 'constant', constant_values = 0)
                # cut if longer than max_time_steps
                if x_entity.shape[0] > args.t_max: x_entity = x_entity[0:args.t_max, :]
                xs_entity.append(x_entity)
            xs_entity = np.array(xs_entity)
            xs_entity = torch.FloatTensor(xs_entity)

            # get squad labels
            y_int, y_atk = [], []
            
            num_squads = len(data_dict[tactic_id][combat_id]['squad_id'])

            for squad in range(num_squads):

                y_int_squad = data_dict[tactic_id][combat_id]['squad_id'][squad]['y_int']
                y_int.append(y_int_squad) 

                y_atk_squad = data_dict[tactic_id][combat_id]['squad_id'][squad]['y_atk']
                y_atk.append(y_atk_squad)
            
            y_int = np.array(y_int)
            y_int = torch.FloatTensor(y_int)

            y_atk = np.array(y_atk)
            y_atk = torch.FloatTensor(y_atk)

            # get entity2squad_idx S_j
            entity2squad_idx = []

            num_entities = len(data_dict[tactic_id][combat_id]['entity_id'])
            for entity in range(num_entities): 
                entity2squad_idx.append([entity, data_dict[tactic_id][combat_id]['entity_id'][entity]['squad']])
            
            entity2squad_idx = np.array(entity2squad_idx)
            entity2squad_idx = torch.LongTensor(entity2squad_idx).T
            entity2squad_idx = F.one_hot(entity2squad_idx[1]).float()
            
            # append to list
            tactic_id_list.append(tactic_id)
            combat_id_list.append(combat_id)
            xs_list.append(xs_entity.unsqueeze(0))
            y_int_list.append(y_int.unsqueeze(0))
            y_atk_list.append(y_atk.unsqueeze(0))
            entity2squad_idx_list.append(entity2squad_idx.unsqueeze(0))

    # convert to tensor
    tactic_id = torch.LongTensor(tactic_id_list)
    combat_id = torch.LongTensor(combat_id_list)
    xs = torch.cat(xs_list, dim = 0)
    y_int = torch.cat(y_int_list, dim = 0)
    y_atk = torch.cat(y_atk_list, dim = 0)
    entity2squad_idx = torch.cat(entity2squad_idx_list, dim = 0)
    squad2squad_idx = torch.ones(max_num_squads, max_num_squads).nonzero().t() 

    # final combat list
    combat_list = [xs, y_int, y_atk, squad2squad_idx, entity2squad_idx, tactic_id, combat_id]

    """
    dimension info
    """
    # x[:,:,0:3] = 'PositionLat(deg)', 'PositionLon(deg)', 'PositionAlt(m)'
    # x[:,:,3] ='AttitudeYaw(deg)'
    # x[:,:,4] = Speed (km/\h)
    # x[:,:,5] = Force Identifier
    # x[:,:,6:] = 'T_Road', 'T_Forest', 'T_OpenLane', 'T_HidingPlace', 'T_Building'

    return combat_list

def data_to_gpu(device, mini_batch, y_int_weight, y_atk_weight, squad2squad_idx):

    features, int_label, atk_label, entity2squad_idx, tactic_id, combat_id = mini_batch

    features = features.to(device)
    int_label = int_label.to(device)
    atk_label = atk_label.to(device)
    entity2squad_idx = entity2squad_idx.to(device)
    squad2squad_idx = squad2squad_idx.to(device)
    
    if y_int_weight is not None: y_int_weight = y_int_weight.to(device)
    if y_atk_weight is not None: y_atk_weight = y_atk_weight.to(device)

    return features, int_label, atk_label, squad2squad_idx, entity2squad_idx, tactic_id, combat_id, y_int_weight, y_atk_weight

def feat_transform(args, x, entity2squad_idx, mean, std):

    xs_entity = x.clone()

    xs_entity_sample = []
    for i in range(xs_entity.shape[0]):
        t_samples = random.sample(range(args.t_max), args.t_sample_size)
        t_samples.sort()
        x_entity = xs_entity[i,:, t_samples, :] 
        xs_entity_sample.append(x_entity.unsqueeze(0))
    xs_entity = torch.cat(xs_entity_sample, dim = 0)

    # find binary features
    binary_feat_idx = []
    for col in range(xs_entity.shape[-1]):
        if len(torch.unique(xs_entity[:,:,:,col])) == 2: binary_feat_idx.append(col)
    binary_feat_idx = torch.LongTensor(binary_feat_idx)
    binary_feat = xs_entity[:,:,:,binary_feat_idx].clone()
    
    # normalize continuous features
    xs_entity = (xs_entity - mean) / std
    xs_entity[:,:,:,binary_feat_idx] = binary_feat
    xs_entity = torch.where(torch.isnan(xs_entity), torch.zeros_like(xs_entity), xs_entity)
    
    # permute dim 
    ### from [num_combat, num_timestep, num_entity, num_feature] 
    ### to [num_combat, num_entity, num_timestep, num_feature]
    xs_entity = xs_entity.permute(0, 2, 1, 3) 

    # stress test
    if args.stress_type == 'noise': xs_entity = add_noise(xs_entity, args.noise)
    elif args.stress_type == 'mask': xs_entity = masking(xs_entity, entity2squad_idx, args.mask_prob, args.mask_type)
    elif args.stress_type == 'none': pass
    else: raise NotImplementedError

    return xs_entity

def data_loader(args, exp, combat_list):

    xs, y_int, y_atk, squad2squad_idx, entity2squad_idx, tactic_id, combat_id = combat_list

    num_tactic = int(tactic_id.max() + 1)
    num_Tc = xs.shape[0]

    # train/val/test indices
    test_idx = [list(range(num_tactic))[0:3][exp // 10]] # do not use last tactic as the test, since it includes unseen y_int
    train_idx = list(range(num_tactic))
    train_idx.remove(test_idx[0])

    test_idx = [i in test_idx for i in tactic_id]
    train_idx = [i in train_idx for i in tactic_id]

    # initialize and normalize 
    mean = torch.mean(xs[train_idx], dim = (0,1,2))
    std = torch.std(xs[train_idx], dim = (0,1,2))
    train_xs = feat_transform(args, xs[train_idx], entity2squad_idx[train_idx], mean, std)
    test_xs = feat_transform(args, xs[test_idx], entity2squad_idx[test_idx], mean, std)
    
    # split
    train_set = [train_xs, y_int[train_idx], y_atk[train_idx], entity2squad_idx[train_idx], tactic_id[train_idx], combat_id[train_idx]]
    test_set = [test_xs, y_int[test_idx], y_atk[test_idx], entity2squad_idx[test_idx], tactic_id[test_idx], combat_id[test_idx]]

    # label weights
    y_int_weight, y_atk_weight = label_weights_generation(y_int[train_idx], y_atk[train_idx])

    # load data in the loader
    train_set = DataLoader(TensorDataset(*train_set), batch_size = args.batch_size, shuffle = False, drop_last = False)
    test_set = DataLoader(TensorDataset(*test_set), batch_size = args.batch_size, shuffle = False, drop_last = False)

    return train_set, test_set, y_int_weight, y_atk_weight, squad2squad_idx

def masking(x, entity2squad_idx, mask_prob, mask_type):
    """
    Masking function
    """
    # x.shape == [num_combat, num_entity, num_timestep, num_feature]

    if mask_type == 'entity': 
        """
        mask entity randomly
        """
        x_mask = torch.rand(x.size(0), x.size(1), x.size(2), 1) > mask_prob 
        x_mask = x_mask > mask_prob
        x_mask = x_mask.repeat(1, 1, 1, x.size(3))
        
        x = x * x_mask # all features are masked, for random entity at random time

    elif mask_type == 'squad': 
        """
        mask squad randomly
        """
        entity2squad_idx = entity2squad_idx.permute(0, 2, 1)
        squad_mask = torch.rand(entity2squad_idx.size(0), x.size(1), entity2squad_idx.size(2)) > mask_prob
        entity_mask = torch.matmul(squad_mask.float(), entity2squad_idx.float()).unsqueeze(3) 
        entity_mask = entity_mask.repeat(1, 1, 1, x.size(3))

        x = x * entity_mask # all features are masked, for random squad at random time

    elif mask_type == 'feat':
        """
        mask feature randomly
        """
        entity_mask = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3)) > mask_prob
        entity_mask = entity_mask > mask_prob

        x = x * entity_mask # random features are masked, for random entity at random time

    elif mask_type == 'time':
        """
        mask time randomly
        """
        time_mask = torch.rand(x.size(0), x.size(1), 1, 1) > mask_prob
        time_mask = time_mask > mask_prob
        time_mask = time_mask.repeat(1, 1, x.size(2), x.size(3))

        x = x * time_mask # all features are masked, for random entity for for all time

    else:
        raise NotImplementedError

    return x

def add_noise(x, noise_level):
    """
    Noising Function 
    """

    assert noise_level in [0.8, 1.6, 2.4, 3.2, 4.0]
    noise_prob = noise_level * 0.1

    # add gaussian noise to continuous features
    gaussian_noise = torch.distributions.normal.Normal(0, noise_level)
    noise = gaussian_noise.sample(x[:,:,:,:5].shape)
    x[:,:,:,:5] = x[:,:,:,:5] + noise
    
    # add bernoulli noise to binary features
    noise_idx = torch.rand(x[:,:,:,5:].shape) > noise_prob
    x[:,:,:,5:] = torch.where(noise_idx, x[:,:,:,5:], 1-x[:,:,:,5:])

    return x

