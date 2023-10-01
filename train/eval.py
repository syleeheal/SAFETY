import pdb
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
from data_utils import data_to_gpu
import logging


def infer(args, test_set, best_model, squad2squad_idx):

    best_model.eval()
    int_probs, atk_probs, int_labels, atk_labels = [], [], [], []
    
    with torch.no_grad(): 
        for test_batch in test_set:
            features, int_label, atk_label, squad2squad_idx, entity2squad_idx, _, _, _, _= data_to_gpu(args.device, test_batch, None, None, squad2squad_idx)
            int_prob, atk_prob = best_model(features, entity2squad_idx, squad2squad_idx)
    
            if args.pred == 'y_atk': 
                atk_probs.append(atk_prob)
                atk_labels.append(atk_label)
            if args.pred == 'y_int': 
                int_probs.append(int_prob)
                int_labels.append(int_label)

    if args.pred == 'y_atk': 
        atk_probs = torch.cat(atk_probs, dim = 0)
        atk_labels = torch.cat(atk_labels, dim = 0)
    if args.pred == 'y_int': 
        int_probs = torch.cat(int_probs, dim = 0)    
        int_labels = torch.cat(int_labels, dim = 0)
        
    return int_probs, atk_probs, int_labels, atk_labels

def eval_atk(exp, atk_prob, atk_label):

    atk_pred = atk_prob.cpu().numpy().flatten()
    atk_label = atk_label.cpu().numpy().flatten()
    auroc = roc_auc_score(atk_label,atk_pred)
    print("[Atk. Pred] AUROC: {:.4f}".format(auroc))
    
    return auroc

def eval_int(exp, int_prob, int_label):

    int_pred = (int_prob > 0.5).cpu().numpy().reshape(-1, int_prob.shape[-1])
    int_label = int_label.cpu().numpy().reshape(-1, int_label.shape[-1])

    f1_macro = f1_score(int_label, int_pred, average='macro', zero_division=0)
    f1_micro = f1_score(int_label, int_pred, average='micro', zero_division=0)
    print("[Int. Pred] F1_Macro: {:.4f}, F1_Micro: {:.4f}".format(f1_macro, f1_micro))

    return f1_macro, f1_micro

def save_results(args, auroc_scores, f1_macro_scores, f1_micro_scores):

    print()
    print('Model: {}'.format(args.model))
    print('Predict: {}'.format(args.pred))
    print('Split: {}'.format(args.split))
    if args.stress_type == 'noise':
        print('Stress Type: {}'.format(args.stress_type))
        print('Noise Level: {}'.format(args.noise))
    if args.stress_type != 'mask':
        print('Stress Type: {}'.format(args.stress_type))
        print('Mask Type: {}'.format(args.mask_type))
        print('Mask Prob: {}'.format(args.mask_prob))
    print()

    if args.pred == 'y_atk': 
        print("Mean AUROC: {:.2f} ± {:.2f}".format(np.mean(auroc_scores)*100, np.std(auroc_scores)*100))

    if args.pred == 'y_int': 
        print("Mean F1-Macro: {:.2f} ± {:.2f}".format(np.mean(f1_macro_scores)*100, np.std(f1_macro_scores)*100))
        print("Mean F1-Micro: {:.2f} ± {:.2f}".format(np.mean(f1_micro_scores)*100, np.std(f1_micro_scores)*100))

    print()

    # log results
    logging.basicConfig(filename='./SAFETY/results/' + args.model + '_' + args.pred + '_' + args.stress_type + '.log', level=logging.INFO)
    
    logging.info('Model: {}'.format(args.model))
    logging.info('Split: {}'.format(args.split))
    logging.info('Predict: {}'.format(args.pred))

    if args.stress_type in ['noise', 'none']:
        logging.info('Stress Type: {}'.format(args.stress_type))
        logging.info('Noise Level: {}'.format(args.noise))
    elif args.stress_type == 'mask':
        logging.info('Stress Type: {}'.format(args.stress_type))
        logging.info('Mask Type: {}'.format(args.mask_type))
        logging.info('Mask Prob: {}'.format(args.mask_prob))
    
    if args.pred == 'y_atk': 
        logging.info("Mean AUROC: {:.2f} ± {:.2f}".format(np.mean(auroc_scores)*100, np.std(auroc_scores)*100))
    if args.pred == 'y_int':
        logging.info("Mean F1-Macro: {:.2f} ± {:.2f}".format(np.mean(f1_macro_scores)*100, np.std(f1_macro_scores)*100))
        logging.info("Mean F1-Micro: {:.2f} ± {:.2f}".format(np.mean(f1_micro_scores)*100, np.std(f1_micro_scores)*100))
    logging.info('\n')
