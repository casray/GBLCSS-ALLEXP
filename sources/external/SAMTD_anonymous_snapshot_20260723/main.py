import warnings
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import yaml
from model.GCN import GCNNet
from model.teachers_weight_matrix import weight_matrix
from model.classifier import classifier
from utils.load_data import load_data
from utils.train_test import train_model, train_s, sim
from utils.add_noisy import noisify_with_P
from utils.label_correction import pre, pre_sim, select_clean, select_pseudo, select_noisy
from encoder.teacher1.get_embed1 import get_embed1
from encoder.teacher2.get_embed2 import get_embed2
from encoder.teacher3.get_embed3 import get_embed3


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arg_parse(dataset_name, noise_type, ites, pseudon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='FewGraph arguments.')
    parser.add_argument('--dataset_name', type=str, help='Cora/CiteSeer/Pubmed/Photo/dblp'),
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--epoch', type=int, help='max epoch')
    parser.add_argument('--no_progress', type=int, help='How many epochs no progress then break training')
    parser.add_argument('--noisy_rate', type=float, help='noisy rate')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--pseudo_num', type=int,
                        help='How many pseudo tags are selected for each category in each round')
    parser.add_argument('--device', type=str, help='cuda or cpu')
    parser.add_argument('--iteration', type=int, help='iteration rounds')
    parser.add_argument('--noise_type', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')
    parser.add_argument('--noise_seed', type=int, default=0)
    parser.add_argument('--noise_rate', type=list, default=[0])
    parser.add_argument('--Matrix_lr', type=float, default=0.01)

    parser.set_defaults(
        dataset_name=dataset_name,
        epoch=1000,
        no_progress=100,
        noisy_rate=0,
        random_seed=0,
        pseudo_num=pseudon,
        device=device,
        noise_type=noise_type,
        iteration=ites,
    )
    return parser.parse_args()


def main(args, noise_r):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args.noisy_rate = noise_r

    '''load data'''
    setup_seed(0)
    dataset, data, idx_train = load_data(args.dataset_name, device=args.device)
    data.to(args.device)

    no_train = data.val_mask + data.test_mask
    Partial_mask = (no_train == 0).bool()

    '''add noise'''
    nclass = dataset.num_classes
    train_labels = data.y[idx_train].cpu().numpy()
    noise_y = noisify_with_P(train_labels, nclass, args.noisy_rate, 10, args.noise_type)
    noise_y = torch.tensor(noise_y)
    data.y[idx_train] = noise_y.to(args.device)

    setup_seed(args.random_seed)

    '''student model'''
    student = GCNNet(dataset).to(args.device)
    optimizer_student = torch.optim.Adam(student.parameters(),
                                         lr=0.001 if args.dataset_name in ['Photo', 'Computers'] else 0.01)

    '''teacher moders'''
    classfier_teacher1 = classifier(embeds_1.shape[1], dataset.num_classes).to(args.device)
    optimizer_teacher1 = torch.optim.Adam(classfier_teacher1.parameters(), lr=0.001 if args.dataset_name in ['Photo', 'Computers'] else 0.01)
    classfier_teacher2 = classifier(embeds_2.shape[1], dataset.num_classes).to(args.device)
    optimizer_teacher2 = torch.optim.Adam(classfier_teacher2.parameters(), lr=0.001 if args.dataset_name in ['Photo', 'Computers'] else 0.01)
    classfier_teacher3 = classifier(embeds_3.shape[1], dataset.num_classes).to(args.device)
    optimizer_teacher3 = torch.optim.Adam(classfier_teacher3.parameters(), lr=0.001 if args.dataset_name in ['Photo', 'Computers'] else 0.01)

    best_acc = 0
    for iters in range(args.iteration):

        '''train teacheers'''
        classfier_teacher1 = train_model(classfier_teacher1, 'teacher1', embeds_1, data, optimizer_teacher1)
        classfier_teacher2 = train_model(classfier_teacher2, 'teacher2', embeds_2, data, optimizer_teacher2)
        classfier_teacher3 = train_model(classfier_teacher3, 'teacher3', embeds_3, data, optimizer_teacher3)

        pre_teacher1, pre_label1 = pre(classfier_teacher1, embeds_1, data)
        pre_teacher2, pre_label2 = pre(classfier_teacher2, embeds_2, data)
        pre_teacher3, pre_label3 = pre(classfier_teacher3, embeds_3, data)

        pre_add = F.softmax(pre_teacher1 + pre_teacher2 + pre_teacher3, dim=1)

        sample_weight = torch.diagonal(pre_sim(pre_teacher1, pre_add) + pre_sim(pre_teacher2, pre_add) + pre_sim(pre_teacher3, pre_add))

        pre_label_add = pre_add.max(1)[1]
        add_acc = pre_label_add[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        teacher1_acc = pre_teacher1[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        teacher2_acc = pre_teacher2[data.test_mask].max(1)[1].eq(
            data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        teacher3_acc = pre_teacher3[data.test_mask].max(1)[1].eq(
            data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print("pre_add: ", add_acc)
        print(teacher1_acc, teacher2_acc, teacher3_acc)

        student, acc = train_s(student, data, optimizer_student, sim_add, Partial_mask, pre_add, sample_weight)
        print("student_acc: ", acc)

        if iters != args.iteration - 1:
            remove_select = select_noisy(args, data, student)
            for i in remove_select:
                data.train_mask[i] = 0

            labels_idx_S, labels_idx_T, label_idex = select_pseudo(student(data), pre_add, data, args)

            for i in range(len(label_idex)):
                for j in label_idex[i]:
                    data.train_mask[j] = 1
                    data.y[j] = i

    student.eval()
    pre_test = student(data)[data.test_mask]
    test_acc = pre_test.max(1)[1].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print("Student_test_acc: ", test_acc)
    Student_ACC.append(torch.tensor(test_acc))



warnings.filterwarnings('ignore')
torch.cuda.set_device(0)
for dat in ['Cora', 'CiteSeer', 'Pubmed','dblp', 'Photo', ]:
    args = arg_parse(dat, noise_type='uniform')  # 'uniform', 'pair'
    with open('utils/args.yaml', 'r') as file:
        parms_all = yaml.load(file, Loader=yaml.FullLoader)
    parms = parms_all[args.dataset_name]
    args.noise_seed = parms['noise_seed']
    for key in parms[args.noise_type].keys():
        setattr(args, key, parms[args.noise_type][key])

    embeds_1 = get_embed1(args).to(args.device)
    embeds_2 = get_embed2(args).to(args.device)
    embeds_3 = get_embed3(args).to(args.device)

    sim_teacher1 = sim(embeds_1)
    sim_teacher2 = sim(embeds_2)
    sim_teacher3 = sim(embeds_3)
    sim_add = sim_teacher1 + sim_teacher2 + sim_teacher3

    for n_rate in [0.2, 0.4, 0.6]:
        Student_ACC = []
        print("noise rate:", n_rate)
        for k in range(5):
            print("seed:", k)
            args.random_seed = k
            main(args, n_rate)
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
                print(f"[GPU] Peak memory: {max_mem:.2f} MB")

        Student_ACC = torch.stack(Student_ACC)
        print("Acc of Our: {:.4f}, std:{:.4f}, noise_rate: {}\n".format(Student_ACC.mean().item() * 100,
                                                                        Student_ACC.std().item() * 100,
                                                                                n_rate), Student_ACC)



