import argparse
import importlib
import nni
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.labelnoise import label_process


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
    'nrgnn_ballaux': 'predictor.NRGNN_Predictor',
    'nrgnn_ballaux_shuffled': 'predictor.NRGNN_Predictor',
    'nrgnn_ballaux_l001': 'predictor.NRGNN_Predictor',
    'nrgnn_ballaux_shuffled_l001': 'predictor.NRGNN_Predictor',
    'nrgnn_ballaux_l01': 'predictor.NRGNN_Predictor',
    'nrgnn_ballaux_shuffled_l01': 'predictor.NRGNN_Predictor',
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
    'frozengrace_gb_gcn': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_gated': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_gated_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_soft': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_soft_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_xh': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_xh_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_xh_topoalpha': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_xh_topoalpha_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'gcn_xh_ce': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_dynalpha': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_dynalpha_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_dynalpha_fast': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_dynalpha_fast_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_bc': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_bc_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_expand_l005': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_softdist_expand_l01': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_adaptive': 'predictor.FrozenGraceGB_GCN_Predictor',
    'frozengrace_gb_gcn_adaptive_shuffled': 'predictor.FrozenGraceGB_GCN_Predictor',
}


def load_predictor(method):
    module = importlib.import_module(PREDICTOR_MODULES[method])
    return getattr(module, method + '_Predictor')


def merge_params(model_conf):
    turner_params = nni.get_next_parameter()
    print(turner_params)
    for item in turner_params.keys():
        print(item)
        if item in ['lr', 'weight_decay']:
            model_conf.training[item] = turner_params[item]
        else:
            model_conf.model[item] = turner_params[item]
    print(model_conf)
    return model_conf


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='dblp',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                             'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='gcn',
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching',
                             'apl', 'sce', 'forward', 'backward', 'lcat', 'mlp',
                             'nrgnn', 'nrgnn_ballaux', 'nrgnn_ballaux_shuffled',
                             'nrgnn_ballaux_l001', 'nrgnn_ballaux_shuffled_l001',
                             'nrgnn_ballaux_l01', 'nrgnn_ballaux_shuffled_l01',
                             'rtgnn', 'cp', 'unionnet', 'cgnn', 'tss',
                             'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn', 'r2lp',
                             'orgingb_gcn', 'orgingb_gcn_random',
                             'orgingb_gcn_reweight', 'orgingb_gcn_reweight_random',
                             'orgingb_gcn_reweight_noball',
                             'reliablegb_gcn_reweight', 'reliablegb_gcn_reweight_random',
                             'reliablegb_gcn_reweight_noball',
                             'frozengrace_gb_gcn', 'frozengrace_gb_gcn_shuffled',
                             'frozengrace_gb_gcn_gated', 'frozengrace_gb_gcn_gated_shuffled',
                             'frozengrace_gb_gcn_soft', 'frozengrace_gb_gcn_soft_shuffled',
                             'frozengrace_gb_gcn_softdist', 'frozengrace_gb_gcn_softdist_shuffled',
                             'frozengrace_gb_gcn_softdist_xh',
                             'frozengrace_gb_gcn_softdist_xh_shuffled',
                             'frozengrace_gb_gcn_softdist_xh_topoalpha',
                             'frozengrace_gb_gcn_softdist_xh_topoalpha_shuffled',
                             'gcn_xh_ce',
                             'frozengrace_gb_gcn_softdist_dynalpha',
                             'frozengrace_gb_gcn_softdist_dynalpha_shuffled',
                             'frozengrace_gb_gcn_softdist_dynalpha_fast',
                             'frozengrace_gb_gcn_softdist_dynalpha_fast_shuffled',
                             'frozengrace_gb_gcn_softdist_bc',
                             'frozengrace_gb_gcn_softdist_bc_shuffled',
                             'frozengrace_gb_gcn_softdist_expand_l005',
                             'frozengrace_gb_gcn_softdist_expand_l01',
                             'frozengrace_gb_gcn_adaptive',
                             'frozengrace_gb_gcn_adaptive_shuffled'],
                    help="Select methods")
parser.add_argument('--noise_type', type=str,
                    default='instance',
                    choices=['clean', 'uniform', 'pair', 'random', 'instance'], help='Type of label noise')
parser.add_argument('--noise_rate', type=float,
                    default='0.3',
                    help='Label noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:0',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000,
                    help="Random Seed")
parser.add_argument('--split_type', type=str,
                    default=None,
                    choices=['default', 'percent', 'samples_per_class'],
                    help='Override dataset split type')
parser.add_argument('--train_percent', type=float,
                    default=None,
                    help='Override training split percentage')
parser.add_argument('--val_percent', type=float,
                    default=None,
                    help='Override validation split percentage')
parser.add_argument('--test_percent', type=float,
                    default=None,
                    help='Override test split percentage')
args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    data_path = './data/'
    data_conf = load_conf('./config/_dataset/' + args.dataset + '.yaml')
    if args.split_type is not None:
        data_conf.split['split_type'] = args.split_type
    for key in ['train_percent', 'val_percent', 'test_percent']:
        value = getattr(args, key)
        if value is not None:
            data_conf.split[key] = value
    if nni.get_trial_id() == "STANDALONE":
        setup_seed(args.seed)
    data = Dataset(args.dataset, path=data_path,
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
    print('Current device: ' + str(data.feats.device))
    model_conf = load_conf(None, args.method, data.name)
    if nni.get_trial_id() != "STANDALONE":
        model_conf = merge_params(model_conf)
    data.noisy_label, modified_mask = label_process(labels=data.labels, features=data.feats,
                                                    n_classes=data.n_classes,
                                                    noise_type=args.noise_type, noise_rate=args.noise_rate,
                                                    random_seed=args.seed, debug=True)

    model_conf.model['n_feat'] = data.dim_feats
    model_conf.model['n_classes'] = data.n_classes
    model_conf.training['debug'] = True
    predictor_cls = load_predictor(args.method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = args.method
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = args.seed
    result = predictor.train()
    if nni.get_trial_id() != "STANDALONE":
        nni.report_final_result(float(result['test']))
