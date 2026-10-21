import argparse
import importlib
import warnings
import time
import pandas as pd
import numpy as np
from utils.labelnoise import label_process
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.logger import MultiExpRecorder, ResultLogger


PREDICTOR_MODULES = {
    'gcn': 'predictor.GCN_Predictor',
    'gin': 'predictor.GIN_Predictor',
    'smodel': 'predictor.Smodel_Predictor',
    'jocor': 'predictor.JoCoR_Predictor',
    'coteaching': 'predictor.Coteaching_Predictor',
    'apl': 'predictor.APL_Predictor',
    'sce': 'predictor.SCE_Predictor',
    'forward': 'predictor.Forward_Predictor',
    'backward': 'predictor.Backward_Predictor',
    'lcat': 'predictor.LCAT_Predictor',
    'mlp': 'predictor.MLP_Predictor',
    'nrgnn': 'predictor.NRGNN_Predictor',
    'rtgnn': 'predictor.RTGNN_Predictor',
    'cp': 'predictor.CP_Predictor',
    'unionnet': 'predictor.UnionNET_Predictor',
    'cgnn': 'predictor.CGNN_Predictor',
    'tss': 'predictor.TSS_Predictor',
    'crgnn': 'predictor.CRGNN_Predictor',
    'clnode': 'predictor.CLNode_Predictor',
    'rncgln': 'predictor.RNCGLN_Predictor',
    'pignn': 'predictor.PIGNN_Predictor',
    'dgnn': 'predictor.DGNN_Predictor',
    'r2lp': 'predictor.R2LP_Predictor',
    'orgingb_gcn': 'predictor.OrginGB_GCN_Predictor',
    'orgingb_gcn_random': 'predictor.OrginGB_GCN_Predictor',
    'orgingb_gcn_reweight': 'predictor.OrginGB_GCN_Predictor',
    'orgingb_gcn_reweight_random': 'predictor.OrginGB_GCN_Predictor',
    'orgingb_gcn_reweight_noball': 'predictor.OrginGB_GCN_Predictor',
    'reliablegb_gcn_reweight': 'predictor.ReliableGB_GCN_Predictor',
    'reliablegb_gcn_reweight_random': 'predictor.ReliableGB_GCN_Predictor',
    'reliablegb_gcn_reweight_noball': 'predictor.ReliableGB_GCN_Predictor',
}


def load_predictor(method):
    module = importlib.import_module(PREDICTOR_MODULES[method])
    return getattr(module, method + '_Predictor')


def run_single_exp(dataset, method_name, seed, noise_type, noise_rate, device, debug=True):
    setup_seed(seed)
    model_conf = load_conf(None, method_name, dataset.name)
    dataset.noisy_label, modified_mask = label_process(labels=dataset.labels, features=dataset.feats,
                                                       n_classes=dataset.n_classes,
                                                       noise_type=noise_type, noise_rate=noise_rate,
                                                       random_seed=seed, debug=debug)
    incorrect_labeled_train_mask = dataset.train_masks[np.isin(dataset.train_masks, modified_mask)]
    correct_labeled_train_mask = dataset.train_masks[~np.isin(dataset.train_masks, modified_mask)]
    supervised_mask = get_neighbors(dataset.adj, dataset.train_masks)
    incorrect_supervised_mask = get_neighbors(dataset.adj, incorrect_labeled_train_mask)
    correct_supervised_mask = get_neighbors(dataset.adj, correct_labeled_train_mask)
    unlabeled_incorrect_supervised_mask = dataset.test_masks[np.isin(dataset.test_masks, incorrect_supervised_mask)]
    unlabeled_correct_supervised_mask = dataset.test_masks[np.isin(dataset.test_masks, correct_supervised_mask)]
    unlabeled_unsupervised_mask = dataset.test_masks[~np.isin(dataset.test_masks, supervised_mask)]

    model_conf.model['n_feat'] = dataset.dim_feats
    model_conf.model['n_classes'] = dataset.n_classes
    model_conf.training['debug'] = debug
    predictor_cls = load_predictor(method_name)
    predictor = predictor_cls(model_conf, dataset, device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method_name
    predictor.noise_type = noise_type
    predictor.noise_rate = noise_rate
    predictor.seed = seed
    original_result = predictor.train()
    extended_result = original_result
    _, correct_labeled_train_accuracy = predictor.test(correct_labeled_train_mask)
    _, incorrect_labeled_train_accuracy = predictor.test(incorrect_labeled_train_mask)
    _, incorrect_labeled_mislead_train_accuracy = predictor.evaluate(
        predictor.noisy_label, incorrect_labeled_train_mask)
    _, unlabeled_unsupervised_accuracy = predictor.test(unlabeled_unsupervised_mask)
    _, unlabeled_correct_supervised_accuracy = predictor.test(unlabeled_correct_supervised_mask)
    _, unlabeled_incorrect_supervised_accuracy = predictor.test(unlabeled_incorrect_supervised_mask)
    extended_result['correct_labeled_train_accuracy'] = correct_labeled_train_accuracy
    extended_result['incorrect_labeled_train_accuracy'] = incorrect_labeled_train_accuracy
    extended_result['incorrect_labeled_mislead_train_accuracy'] = incorrect_labeled_mislead_train_accuracy
    extended_result['unlabeled_correct_supervised_accuracy'] = unlabeled_correct_supervised_accuracy
    extended_result['unlabeled_unsupervised_accuracy'] = unlabeled_unsupervised_accuracy
    extended_result['unlabeled_incorrect_supervised_accuracy'] = unlabeled_incorrect_supervised_accuracy
    extended_result['total_time'] = predictor.total_time

    return original_result, extended_result


parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int,
                    default=10,
                    help="Number of experiments for each combination of method and data")
parser.add_argument('--methods', type=str, nargs='+',
                    default=['gcn', 'smodel', 'coteaching', 'jocor', 'apl', 'sce', 'forward', 'backward'],
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching',
                             'apl', 'sce', 'forward', 'backward', 'lcat', 'mlp',
                             'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'tss',
                             'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn', 'r2lp',
                             'orgingb_gcn', 'orgingb_gcn_random',
                             'orgingb_gcn_reweight', 'orgingb_gcn_reweight_random',
                             'orgingb_gcn_reweight_noball',
                             'reliablegb_gcn_reweight', 'reliablegb_gcn_reweight_random',
                             'reliablegb_gcn_reweight_noball'],
                    help='Select methods')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                             'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                             'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                    help='Select datasets')
parser.add_argument('--noise_type', type=str, nargs='+',
                    default=['clean', 'pair', 'uniform'],
                    choices=['clean', 'pair', 'uniform', 'random', 'instance'], help='Noise type')
parser.add_argument('--noise_rate', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help='Noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:0',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000, help="Random Seed")
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    warnings.filterwarnings("ignore")
    data_path = './data/'
    method_list = args.methods
    data_list = args.datasets
    noise_type_list = args.noise_type
    noise_rate_list = args.noise_rate

    noise_list = []
    for noise_type in noise_type_list:
        if noise_type == 'clean':
            # noise_rate is not available for clean data
            noise_list.append([0.0, noise_type])
            continue
        for noise_rate in noise_rate_list:
            noise_list.append([noise_rate, noise_type])

    result_recorder = ResultLogger(method_list, data_list, noise_list, args.runs)
    for noise_rate, noise_type in noise_list:
        for data_name in data_list:
            setup_seed(args.seed)
            data_conf = load_conf('./config/_dataset/' + data_name + '.yaml')
            data = Dataset(data_name, path=data_path,
                           feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                           train_size=data_conf.split['train_size'],
                           val_size=data_conf.split['val_size'],
                           test_size=data_conf.split['test_size'],
                           train_percent=data_conf.split['train_percent'],
                           val_percent=data_conf.split['val_percent'],
                           test_percent=data_conf.split['test_percent'],
                           train_examples_per_class=data_conf.split['train_examples_per_class'],
                           val_examples_per_class=data_conf.split['val_examples_per_class'],
                           test_examples_per_class=data_conf.split['test_examples_per_class'],
                           add_self_loop=data_conf.modify['add_self_loop'],
                           from_npz=data_conf.modify['from_npz_largest_component'],
                           device=args.device,
                           split_type=data_conf.split['split_type'])

            for method_name in method_list:
                logger = MultiExpRecorder(runs=args.runs)
                for run in range(args.runs):
                    # setup different random seed for each runs
                    setup_seed(args.seed + run)
                    simple_result, total_results = run_single_exp(data, method_name, noise_type=noise_type,
                                                                  noise_rate=noise_rate,
                                                                  seed=args.seed + run, device=args.device, debug=False)
                    logger.add_result(run, total_results)
                total_results = logger.get_statistics()
                result_recorder.dump_record(method_name, data_name, noise_type, noise_rate, total_results)
