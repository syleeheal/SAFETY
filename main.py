import argparse
from data_utils import load_graph_list
from train import Trainer
from train_s import Trainer_S
import pdb


def parameter_parser():
    
    parser = argparse.ArgumentParser()

    """
    data parameters
    """
    parser.add_argument("--num-nodes", type=int, default=54,) # number of entities
    parser.add_argument("--num-units", type=int, default=12,) # number of units
    parser.add_argument("--num-scenarios", type=int, default=4,) # number of tactics
    parser.add_argument("--time-steps", type=int, default=20) # set t_max
    parser.add_argument("--time-init", type=str, default="random",) # how to intialize t


    """
    exp parameters
    """
    parser.add_argument("--model", type=str, default="STA",) # set model \in {STA, RNN, GRU, MLP, P-LSTM}

    parser.add_argument("--device", type=str, default="cuda:0",) # set gpu device
    parser.add_argument("--num-exp", type=int, default=30,) # set number of exps
    parser.add_argument("--num-epochs", type=int, default=100,) # set number of epochs
    parser.add_argument("--patience", type=int, default=100,) # set patience

    parser.add_argument("--split", type=str, default="scenario",) # data split type \in {scenario}

    parser.add_argument("--stress-type", type=str, default="none",) # stress test types \in {none, mask, noise}

    parser.add_argument("--mask-prob", type=float, default=0.5,) # set masking probability \in [0, 1]
    parser.add_argument("--mask-type", type=str, default='combined',) # set masking types \in {feat, node, unit, time, combined}
    parser.add_argument("--noise", type=float, default=0.0,) # set noise standard deviation \in {0.0, 0.8, 1.6, 2.4, 3.2, 4.0}

    parser.add_argument("--pred", type=str, default="joint",) # set downstream task \in {joint, attack, task}


    """
    model parameters
    """
    parser.add_argument("--lr", type=float, default=0.01,) # learning rate
    parser.add_argument("--dr", type=float, default=1e-5,) # learning rate decay

    parser.add_argument("--num-layers", type=int, default=1,) # number of layers

    parser.add_argument("--hid-dim", type=int, default=64,) # hidden dim
    parser.add_argument("--num-heads", type=int, default=1,) # number of heads

    parser.add_argument("--dropout", type=float, default=0.0,) # dropout

    parser.add_argument("--num-estimators", type=int, default=10,) # for xg boost 
    return parser.parse_args()

def best_hyperparameters(args):
    if args.model == 'STA':
        if args.pred == 'attack':
            args.hid_dim = 16
            args.num_heads = 4
            args.dr = 5e-4
            args.dropout = 0.3
            args.num_layers = 1
        if args.pred == 'task':
            args.hid_dim = 16
            args.num_heads = 4
            args.dr = 5e-6
            args.dropout = 0.3
            args.num_layers = 1
    elif args.model == 'GRU':
        if args.pred == 'attack':
            args.dr = 5e-4
            args.dropout = 0.3
            args.num_layers = 2
        if args.pred == 'task':
            args.dr = 1e-4
            args.dropout = 0.3
            args.num_layers = 2
    elif args.model == 'RNN':
        if args.pred == 'attack':
            args.dr = 5e-4
            args.dropout = 0.3
            args.num_layers = 2
        if args.pred == 'task':
            args.dr = 5e-6
            args.dropout = 0.3
            args.num_layers = 2
    elif args.model == 'P-LSTM':
        if args.pred == 'attack':
            args.dr = 5e-04
            args.dropout = 0.0
        if args.pred == 'task':
            args.dr = 5e-6
            args.dropout = 0.3
    elif args.model == 'MLP':
        if args.pred == 'attack':
            args.dr = 5e-4
            args.dropout = 0.5
            args.num_layers = 4
        if args.pred == 'task':
            args.dr = 1e-5
            args.dropout = 0.5
            args.num_layers = 3
    
    return args


def main():

    args = parameter_parser()
    graph_list = load_graph_list(args)

    args = best_hyperparameters(args)
    if args.model in ['STA', 'GRU', 'RNN', 'P-LSTM', 'MLP']:
        Trainer(args, graph_list).fit_model()
    else:
        Trainer_S(args, graph_list).fit_model()

    
    
if __name__ == "__main__":

    main()

