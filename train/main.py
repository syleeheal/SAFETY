import torch
import numpy as np
import random
import argparse
from utils import load_hyperparam
from data_utils import load_combat_list, data_loader
from train import Trainer
from eval import infer, eval_atk, eval_int, save_results
from hyperparam_opt import Hyperparam_Optimizer
import pdb


def parameter_parser():
    
    parser = argparse.ArgumentParser()

    """
    exp param
    """
    parser.add_argument("--device", type=str, default="cuda:0",) # set gpu device
    parser.add_argument("--num-exp", type=int, default=30,) # set number of exps
    parser.add_argument("--num-epochs", type=int, default=50,) # set number of epochs
    parser.add_argument('--patience', type=int, default=10,) # set patience
    parser.add_argument("--batch-size", type=int, default=128,) # set number of epochs
    parser.add_argument("--split", type=str, default="tactic",) # data split type \in {tactic}
    parser.add_argument("--optimize", type=str, default="model",) # optimize type \in {model, hyperparam}
    parser.add_argument("--num-entities", type=int, default=54) # set maximum number of entities

    """
    time param
    """
    parser.add_argument("--t-max", type=int, default=300) # set t_max, the maximum observed timestamp
    parser.add_argument("--t-sample-size", type=int, default=20) # set maximum number of observed timesteps

    """
    stress param
    """
    parser.add_argument("--stress-type", type=str, default="none",) # stress test types \in {none, mask, noise}
    parser.add_argument("--noise", type=float, default=0.0,) # set noise standard deviation \in {0.0, 0.8, 1.6, 2.4, 3.2, 4.0}
    parser.add_argument("--mask-prob", type=float, default=0.0,) # set masking probability \in [0, 1]
    parser.add_argument("--mask-type", type=str, default='none',) # set masking types \in {none, feat, entity, squad, time, combined}

    """
    prediction param
    """
    parser.add_argument("--model", type=str, default="SAFETY",) # set model \in {SAFETY, RNN, GRU, MLP, P-LSTM}
    parser.add_argument("--pred", type=str, default="y_atk",) # set downstream task \in {y_atk, y_int}

    """
    model param
    """
    parser.add_argument("--lr", type=float, default=0.01,) # learning rate
    parser.add_argument("--dr", type=float, default=1e-5,) # learning rate decay
    parser.add_argument("--num-layers", type=int, default=1,) # number of layers
    parser.add_argument("--hid-dim", type=int, default=64,) # hidden dim
    parser.add_argument("--dropout", type=float, default=0.0,) # dropout

    return parser.parse_args()




def main():

    args = parameter_parser()
    seeds = torch.load('./seeds_100.pt')
    torch.cuda.set_device(args.device)
    auroc_scores, f1_micro_scores, f1_macro_scores = [], [], []

    combat_list = load_combat_list(args)

    if args.optimize == 'model':
        
        # load best hyperparam
        args = load_hyperparam(args)
        for exp in range(args.num_exp):

            # fix random seed
            seed = seeds[exp]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # load data
            train_set, test_set, y_int_weight, y_atk_weight, unit2unit_index = data_loader(args, exp, combat_list)

            # train model
            best_model = Trainer(args, combat_list).fit_model(train_set, y_int_weight, y_atk_weight, unit2unit_index)

            # evaluate and save results
            y_int_prob, y_atk_prob, y_int_label, y_atk_label = infer(args, test_set, best_model, unit2unit_index)

            if args.pred == 'y_atk': 
                auroc = eval_atk(exp, y_atk_prob, y_atk_label)
                auroc_scores.append(auroc)

            if args.pred == 'y_int': 
                f1_macro, f1_micro = eval_int(exp, y_int_prob, y_int_label)
                f1_micro_scores.append(f1_micro)
                f1_macro_scores.append(f1_macro)

        save_results(args, auroc_scores, f1_macro_scores, f1_micro_scores)

    elif args.optimize == 'hyperparam':
        Hyperparam_Optimizer(args, combat_list).search()

if __name__ == "__main__":

    main()

