import torch.nn.functional as F
import torch
import torch.nn as nn
import math

def calc_wasserstein_loss(z):
    N = z.size(0)
    D = z.size(1)

    z_center = torch.mean(z, dim=0, keepdim=True)
    mean = z.mean(0)
    covariance = torch.mm((z-z_center).t(), z-z_center)/N + 1e-12*torch.eye(D).cuda()
    #############calculation of part1
    part1 = torch.sum(torch.multiply(mean, mean))

    ######################################################
    S, Q = torch.linalg.eigh(covariance)
    # S = torch.abs(S[:,0])
    S = torch.abs(S)
    mS = torch.sqrt(torch.diag(S))
    covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

    #############calculation of part2
    part2 = torch.trace(covariance - 2.0/math.sqrt(D)*covariance2)
    wasserstein_loss = torch.sqrt(part1+1+part2)

    return wasserstein_loss

def sim(z1: torch.Tensor):
    z1 = F.normalize(z1)
    return torch.mm(z1, z1.t())

def train(model, embs, data, optimizer):
    optimizer.zero_grad()
    loss = F.cross_entropy(F.softmax(model(embs)[data.train_mask], dim=1), data.y[data.train_mask])
    # print("train loss: {}".format(loss))
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss

def val(model, embs, data):
    model.eval()
    logits = model(embs)
    pred = logits[data.val_mask].max(1)[1]
    acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    return acc

def test(model, embs, data):
    model.eval()
    logits, accs = model(embs), []
    pred = logits[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def train_model(model, name, embs, data, optimizer, patient = 100):
    model.train()
    best_acc = 0
    best_loss = 0
    for epoch in range(100000):
        loss_now = train(model, embs, data, optimizer)
        acc_now = val(model, embs, data)
        if acc_now > best_acc:
            best_acc = acc_now
            bad_counter = 0
            torch.save(model.state_dict(), 'best_model_{}.pkl'.format(name))
        else:
            bad_counter += 1
        if bad_counter == patient:
            break
    model.load_state_dict(torch.load('best_model_{}.pkl'.format(name)))
    return model

def train_s(model, data, optimizer, teacher_sim, Partial_mask, pre_add, weight):
    best_acc_val = 0
    bad_counter = 0
    sample_weight = weight.detach()
    for epoch in range(100000):
        model.train()
        optimizer.zero_grad()
        pre = F.log_softmax(model(data), dim=1)

        loss_ce = F.nll_loss(pre[data.train_mask], data.y[data.train_mask])

        soft_label = pre_add.detach()
        loss_soft = (-(soft_label[Partial_mask] * pre[Partial_mask]).sum(dim=1) * sample_weight[Partial_mask]).mean()
        # loss_soft = (-(soft_label[Partial_mask] * pre[Partial_mask]).sum(dim=1)).mean()

        embed = model.embedding(data)
        student_sim = sim(embed)
        student_sim = F.normalize(student_sim, p=2, dim=-1)
        teacher_sim = F.normalize(teacher_sim, p=2, dim=-1)
        loss_encoder = F.l1_loss(student_sim, teacher_sim)

        # loss_calc = calc_wasserstein_loss(embed)

        loss = loss_ce + 15 * loss_soft + epoch * loss_encoder

        loss.backward()
        optimizer.step()

        model.eval()
        pre_val = model(data)
        acc = pre_val[data.val_mask].max(1)[1].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        # print("val_acc:", acc)

        # print(acc_now)
        if acc > best_acc_val:
            best_acc_val = acc
            bad_counter = 0
            torch.save(model.state_dict(), 'best_student.pkl')
            # print(epoch, best_acc_val)
        else:
            bad_counter += 1
        if bad_counter == 100:
            break

    model.load_state_dict(torch.load('best_student.pkl'))
    model.eval()
    pre_test = model(data)[data.test_mask]
    test_acc = pre_test.max(1)[1].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    return model, test_acc

def train_student(model, name, data, teacher_pre, optimizer, clean, patient = 100):
    best_acc = 0
    train_mask = []
    for i in range(data.size(0)):
        if data.val_mask[i] == 0 and data.test_mask[i] == 0:
            train_mask.append(i)
    for epoch in range(100000):
        optimizer.zero_grad()
        # loss = train_s(model, data, teacher_pre, train_mask)
        loss = -(teacher_pre[train_mask] * F.log_softmax(model(data)[train_mask], dim=1)).mean()
        loss.backward(retain_graph=True)
        optimizer.step()


        model.eval()
        logits = model(data)
        pred = logits[data.val_mask].max(1)[1]
        acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        acc_now = acc

        if acc_now > best_acc:
            best_acc = acc_now
            best_loss = loss
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), 'best_model_{}.pkl'.format(name))
            # print(epoch, acc_now)
        else:
            bad_counter += 1
        if bad_counter == patient:
            break
    model.load_state_dict(torch.load('best_model_{}.pkl'.format(name)))
    model.eval()
    logits, accs = model(data), []
    pred = logits[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print("{} test acc: {}".format(name, acc))
    return model, acc, best_acc
