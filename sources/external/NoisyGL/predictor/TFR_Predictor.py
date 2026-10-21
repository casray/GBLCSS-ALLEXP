import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, subgraph

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from predictor.module.TFR import TFRDecoder


class tfr_Predictor(Predictor):
    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                         dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.decoder = TFRDecoder(in_channels=conf.model['n_classes'], hidden_channels=conf.model['n_hidden'],
                                  out_channels=conf.model['n_feat'],
                                  decoder_dropout=conf.model['decoder_dropout'],
                                  att_softmax=conf.model['att_softmax']).to(self.device)
        self.encoder = torch.nn.Linear(in_features=conf.model['n_feat'],
                                       out_features=conf.model['n_hidden'],
                                       bias=True).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

        self.pseudo_coef = 0.0
        self.pseudo_ratio = conf.model['pseudo_ratio']
        self.beta = conf.model['beta']
        self.beta_dropout_ratio = conf.model['beta_dropout_ratio']
        self.beta_dropout_type = conf.model['beta_dropout_type']
        self.use_recon_softmax = conf.model['use_recon_softmax']

    def _dropout_edge_index(self, adj):
        edge_index = adj.indices()
        if self.beta_dropout_type == 'edge':
            return dropout_edge(edge_index, p=self.beta_dropout_ratio)[0]
        subset = torch.randperm(self.n_nodes, device=self.device)[:int(self.n_nodes * self.beta_dropout_ratio)]
        return subgraph(subset=subset, edge_index=edge_index)[0]

    def get_prediction(self, features, adj, label=None, mask=None, training=True):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            pseudo_label = torch.softmax(output, dim=1)
            dropout_adj = self._dropout_edge_index(adj)
            recon_feats, confidence = self.decoder(pseudo_label, dropout_adj)
            high_confidence_indices = torch.argsort(confidence.squeeze(), descending=True)[
                                      :int(self.n_nodes * self.pseudo_ratio)]
            if self.use_recon_softmax:
                recon_feats = F.softmax(recon_feats, dim=1)
            loss_recon = self.loss_fn(recon_feats, features)
            loss_pseudo = self.loss_fn(output[high_confidence_indices],
                                       pseudo_label[high_confidence_indices].detach())
            loss_sup = self.loss_fn(output[mask], label[mask])
            loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon + self.pseudo_coef * loss_pseudo
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            if training:
                self.pseudo_coef = acc
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.decoder.train()
            self.encoder.train()
            self.optim.zero_grad()

            output, loss_train, acc_train = self.get_prediction(
                self.feats, self.adj, self.noisy_label, self.train_mask, training=True)
            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.decoder.eval()
        self.encoder.eval()
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, self.adj, label, mask, training=False)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        return self.evaluate(self.clean_label, mask)


class tfr_gcn_Predictor(tfr_Predictor):
    pass
