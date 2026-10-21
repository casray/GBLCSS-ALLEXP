import torch
import torch.nn.functional as F
import numpy as np



def sim(z1: torch.Tensor):
    z1 = F.normalize(z1)
    return torch.mm(z1, z1.t())

def pre_sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def pre(model, embs, data):
    model.eval()
    pred = F.softmax(model(embs), dim=1)
    pred_label = pred.max(1)[1]
    return pred, pred_label

def select_clean(data, y_teacher_1, y_teacher_2, y_teacher_3, sim_add, teachers_add_pre):
    prelabel_idx = []
    prelabel_idx_sim = []
    prelabel_sim = []
    select_clean_idx = []
    train_idx = []
    train_idx_loss = []
    train_loss = []
    for i in range(max(data.y) + 1):
        prelabel_idx.append([])
        prelabel_idx_sim.append([])
        prelabel_sim.append([])
        select_clean_idx.append([])
        train_idx.append([])
        train_idx_loss.append([])
        train_loss.append([])

    for i in range(len(y_teacher_1)):
        if y_teacher_1[i] == y_teacher_2[i] == y_teacher_3[i] and data.val_mask[i] == 0 and data.test_mask[i] == 0:
            prelabel_idx[y_teacher_1[i]].append(i)

    for i in range(len(prelabel_idx)):
        prelabel_sim[i] = sim_add[prelabel_idx[i]][:, prelabel_idx[i]]
        a = torch.zeros(prelabel_sim[i].shape)
        for j in range(len(a)):
            for k in range(len(a)):
                if j != k:
                    a[j][k] = 1
        prelabel_sim[i] = prelabel_sim[i] * a.cuda()
        for j in range(len(prelabel_sim[i])):
            values, indices = torch.topk(prelabel_sim[i][j],
                                         30 if len(prelabel_sim[i]) > 30 else len(prelabel_sim[i]))
            prelabel_idx_sim[i].append(values.sum().item())
        value, indice = torch.topk(torch.tensor(prelabel_idx_sim[i]),
                                   20 if len(prelabel_sim[i]) > 20 else len(prelabel_sim[i]))
        indice = indice.tolist()
        select_clean_idx[i] = [prelabel_idx[i][k] for k in indice]

    for i in range(len(data.train_mask)):
        if data.train_mask[i] == 1:
            train_idx[data.y[i]].append(i)

    for i in range(max(data.y) + 1):
        for j in train_idx[i]:
            loss = F.cross_entropy(teachers_add_pre[j], data.y[j])
            train_idx_loss[i].append([j, loss])
            train_loss[i].append(loss)
    for i in range(len(train_loss)):
        train_loss[i].sort()

    loss_high = []
    for i in range(len(train_loss)):
        if len(train_loss[i]) == 0:
            continue
        loss_high.append(train_loss[i][int(len(train_loss[i]) * 0.5)])
    for i in range(len(train_idx_loss)):
        for j in train_idx_loss[i]:
            if j[1] < loss_high[i]:
                select_clean_idx[data.y[j[0]]].append(j[0])
    for i in range(len(select_clean_idx)):
        select_clean_idx[i] = list(set(select_clean_idx[i]))

    return prelabel_idx, prelabel_idx_sim, prelabel_sim, select_clean_idx


def select_pseudo(student_pre, teachers_add_pre, data, args):
    No_pseudo = []
    for i in range(len(data.train_mask)):
        if data.val_mask[i] == 1 or data.test_mask[i]==1 or data.train_mask[i]== 1:
            No_pseudo.append(i)

    all_pre = student_pre + teachers_add_pre

    teachers_label = teachers_add_pre.argmax(dim=1)
    student_label = student_pre.argmax(dim=1)
    confidence_student = torch.zeros((data.num_nodes, 1), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    confidence_teachers = torch.zeros((data.num_nodes, 1), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    teachers_label = teachers_label.reshape(len(teachers_label), 1)
    student_label = student_label.reshape(len(student_label), 1)
    for i in range(len(teachers_label)):
        confidence_teachers[i] = teachers_add_pre[i][teachers_label[i]]
        confidence_student[i] = student_pre[i][student_label[i]]

    # number, confidence, pseudo-label
    linshi_S = np.arange(len(data.y))
    linshi_S = linshi_S[:, np.newaxis]
    linshi_T = np.arange(len(data.y))
    linshi_T = linshi_T[:, np.newaxis]
    new_x_S = np.hstack([linshi_S, confidence_student.detach().cpu().numpy()])
    new_x_S = np.hstack([new_x_S, student_label.detach().cpu().numpy()])  # # new_x: [[idx, confidence, pre_label], [idx, confidence, pre_label],...]]
    new_x_T = np.hstack([linshi_T, confidence_teachers.detach().cpu().numpy()])
    new_x_T = np.hstack([new_x_T, teachers_label.detach().cpu().numpy()])  # # new_x: [[idx, confidence, pre_label], [idx, confidence, pre_label],...]]


    labels_mx_S = []
    labels_mx_T = []


    for i in range(max(data.y+1)):
        mx_S = new_x_S[new_x_S[:, -1] == i, :]  # mx:[[[idx, confidence, pre_label == i], [idx, confidence, pre_label == i],...]], [[idx, confidence, pre_label == i+1], [idx, confidence, pre_label == i+1],...]],.....]
        mx_S = torch.from_numpy(mx_S)
        labels_mx_S.append(mx_S[torch.argsort(mx_S[:, -2], descending=True)])

        mx_T = new_x_T[new_x_T[:, -1] == i, :]
        mx_T = torch.from_numpy(mx_T)
        labels_mx_T.append(mx_T[torch.argsort(mx_T[:, -2], descending=True)])

    pseudo_num = args.pseudo_num
    label_idx_S = []
    label_idx_T = []
    for i in range(max(data.y+1)):
        label_idx_S.append([])
        if labels_mx_S[i].shape[0] == 0:
            continue
        for j in range(labels_mx_S[i].shape[0]):
            if len(label_idx_S[i]) == pseudo_num:
                break
            if int(labels_mx_S[i][j, 0]) not in No_pseudo:
                label_idx_S[i].append(int(labels_mx_S[i][j, 0]))
        label_idx_S[i] = sorted(label_idx_S[i])

    for i in range(max(data.y + 1)):
        label_idx_T.append([])
        if labels_mx_T[i].shape[0] == 0:
            continue
        for j in range(labels_mx_T[i].shape[0]):
            if len(label_idx_T[i]) == pseudo_num:
                break
            if int(labels_mx_T[i][j, 0]) not in No_pseudo:
                label_idx_T[i].append(int(labels_mx_T[i][j, 0]))
        label_idx_T[i] = sorted(label_idx_T[i])

    num_idx = 0
    num_pseudo = []
    labels_idx = []
    for i in range(len(label_idx_T)):
        union = set(label_idx_T[i]).union(label_idx_S[i])
        labels_idx.append(sorted(list(union)))
        num_pseudo = set(num_pseudo).union(labels_idx[i])
        num_idx += len(labels_idx[i])

    remove = []
    if len(num_pseudo) != num_idx:
        for i in range(len(labels_idx)):
            for j in range(i + 1, len(labels_idx)):
                for k in labels_idx[i]:
                    if k in labels_idx[j]:
                        remove.append(k)

    for i in remove:
        for j in range(len(labels_idx)):
            if i in labels_idx[j]:
                labels_idx[j].remove(i)

    return label_idx_S, label_idx_T, labels_idx

def select_noisy(args, data, student_model):
    student_pre = F.softmax(student_model(data), dim = 1)

    label_idx = []
    label_idx_loss = []
    label_loss = []
    for i in range(max(data.y)+1):
        label_idx.append([])
        label_idx_loss.append([])
        label_loss.append([])


    for i in range(len(data.train_mask)):
        if data.train_mask[i] == 1:
            label_idx[data.y[i]].append(i)


    for i in range(max(data.y)+1):
        for j in label_idx[i]:
            loss = F.cross_entropy(student_pre[j], data.y[j])
            label_idx_loss[i].append([j, loss])
            label_loss[i].append(loss)
    for i in range(len(label_loss)):
        label_loss[i].sort()


    loss_high = []
    remove_select = []
    a = 0.9
    if args.noisy_rate>0.4:
        a = 0.8
    for i in range(len(label_loss)):
        if len(label_loss[i]) == 0:
            continue
        loss_high.append(label_loss[i][int(len(label_loss[i])*a)])
    for i in range(len(label_idx_loss)):
        for j in label_idx_loss[i]:
            if j[1]>loss_high[i]:
                remove_select.append(j[0])

    return remove_select
