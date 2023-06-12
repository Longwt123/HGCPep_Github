import os
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dhg import Hypergraph
from torch.utils.tensorboard import SummaryWriter
from model.HGCPep_Model import HGCPep_Model
from model.loss_function.focalloss import FocalLoss
from utils.util_data import readDatasetFromPickle
from utils.util_draw import visualize, visualize_LossGraph
from utils.util_metrics import evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)
class_num = config['class_num']


def naive_loss(y_pred, y_true, idx, loss_weight=None, ohem=False, focal=False, gamma=0):  # ohem=True 原来是False
    # 用nn库函数计算loss
    loss_Fun = nn.CrossEntropyLoss()
    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]
    k = 0.7

    loss_output = torch.zeros(num_examples).cuda()
    for i in range(num_task):
        if loss_weight is not None:
            if focal:
                loss_temp = FocalLoss(gamma=gamma)(y_pred[i][idx], y_true[:, i].long())
            else:
                loss_temp = loss_Fun(y_pred[i][idx], y_true[:, i].long())
            out = loss_weight[i] * loss_temp
            loss_output += out
        else:
            if focal:
                loss_temp = FocalLoss(gamma=gamma)(y_pred[i][idx], y_true[:, i].long())
            else:
                loss_temp = loss_Fun(y_pred[i][idx], y_true[:, i].long())
            loss_output += loss_temp

    # Online Hard Example Mining
    if ohem:
        val, idx = torch.topk(loss_output, int(k * num_examples))
        loss_output[loss_output < val[-1]] = 0

    loss = torch.sum(loss_output)

    return loss

def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()


def copyfile(srcfile, path):
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(srcfile, os.path.join(path, fname))


class Trainer(object):
    def __init__(self, config, file_path):
        self.config = config
        self.log_dir = config['log_dir'] + '/{}/{}_{}+{}_{}_{}_{}_{}'.format(
            'finetune', datetime.now().strftime('%b%d_%H-%M'), config['frontPart'],
            config['latterPart'], config['loss_type'],
            config['optim']['init_lr'], config['task_name'], config['epochs']
        )
        self.choose_model = config['choose_model']

        self.alpha = config['alpha']
        self.gamma = config['gamma']

        self.net = self._get_net()
        self.criterion = self._get_loss_fn()
        self.optim = self._get_optim()

        self.data = readDatasetFromPickle(config)
        self.device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.X = None
        self.G = None
        self.lbls = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        self.best_val_importantValue = None
        self.best_test_importantValue = None

        self.pep_seq_raw = None

        if config['naive_loss']['loss_weight']:
            p = None
            if config['class_num'] == 15:
                p = torch.tensor(
                    [0.044659022, 0.038106733, 0.101560479, 0.092680403, 0.178291232, 0.039658591, 0.078023968,
                     0.04362445, 0.021036296, 0.141908785, 0.022243297, 0.08862833, 0.04759031, 0.017242866,
                     0.044745237])
                p = p.cuda()
            if config['class_num'] == 10:
                p = torch.tensor(
                    [0.044659022, 0.101560479, 0.092680403, 0.178291232, 0.078023968, 0.04362445, 0.141908785,
                     0.08862833, 0.04759031, 0.044745237])
                p = p.cuda()
            self.naive_loss_weight = p
        else:
            self.naive_loss_weight = None
        self.naive_ohem = config['naive_loss']['ohem']
        self.naive_focal = config['naive_loss']['focal']
        self.change_status = False  # 是否遇到了最优值
        self.draw_data = config['draw_data']
        self.draw_data['time'] = datetime.now().strftime('%b%d %H_%M')
        file_name = f'PcaAndtSNE_' \
                    f'NO{self.draw_data["draw_class"]}class' \
                    f'_{self.config["frontPart"]}+{self.config["latterPart"]}_' \
                    f'{self.config["loss_type"]}/'
        self.draw_data['dirs'] = os.path.join(self.log_dir, 'visualization', file_name)
        if config['checkpoint']:
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_metric = -np.inf if config['task'] == 'classification' else np.inf
            self.writer = SummaryWriter('{}/{}_{}+{}_{}_{}_{}_{}'.format(
                'finetune', datetime.now().strftime('%b%d_%H-%M'), config['frontPart'],
                config['latterPart'], config['loss_type'],
                config['optim']['init_lr'], config['task_name'], config['epochs']
            ))
        self.txtfile = os.path.join(self.log_dir, 'record.txt')
        copyfile(file_path, self.log_dir)

    def _get_net(self):
        if self.choose_model == 'HGNNP':
            model = HGCPep_Model(self.config)
        else:
            model = HGCPep_Model(self.config)
        return model

    def _get_Embedding(self, input):
        # one hot
        # embedding = torch.eye(num_vertices)
        # text cnn
        embedding = input
        return embedding

    def _get_loss_fn(self):
        loss_type = self.config['loss_type']
        if loss_type == 'yuan':
            return nn.MultiLabelSoftMarginLoss(reduction='mean')
        elif loss_type == 'BCEWithLogitsLoss':
            p = torch.tensor([0.178, 0.078, 0.102, 0.142, 0.044, 0.040, 0.038,
                              0.048, 0.045, 0.093, 0.021, 0.089, 0.022, 0.017, 0.045])
            p = p.cuda()
            return nn.BCEWithLogitsLoss(pos_weight=1 / p)
        elif loss_type == 'CE':
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()

    def _get_optim(self):
        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = eval(self.config['optim']['weight_decay'])
        layer_list = []
        for name, param in self.net.named_parameters():
            if 'mlp_proj' in name:
                layer_list.append(name)
        params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, self.net.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, self.net.named_parameters()))))
        model_params = [{'params': base_params, 'lr': self.config['optim']['init_base_lr']}, {'params': params}]

        if optim_type == 'adam':
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'rms':
            return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'sgd':
            return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('not supported optimizer `!')

    def _get_lr_scheduler(self):
        scheduler_type = self.config['lr_scheduler']['type']
        init_lr = self.config['lr_scheduler']['start_lr']
        warm_up_epoch = self.config['lr_scheduler']['warm_up_epoch']

    def _train_step(self, X, A, lbls, train_idx):
        self.net.train()
        train_loss = 0
        self.optim.zero_grad()
        '''计算'''
        outs, embbeding_y = self.net(X, A)  # outs: list, 15*4019*2
        outs_forNavieLoss = outs
        labels = lbls[train_idx]  # 4019*15 -> 3215*15
        labels = labels.to(torch.float)
        labels_forNavieLoss = labels
        '''展开'''
        # 将outs转换为 48225*2
        temp = outs[0][np.asarray(train_idx)]  # temp: 3215*2
        for i in range(class_num - 1):
            temp = torch.concat((temp, outs[i + 1][np.asarray(train_idx)]), dim=0)
        outs = temp  # 48225*2      48225 = 3215*15
        # 将label转换为 48225*1
        labels = torch.reshape(torch.transpose(labels, 0, 1), (-1,))  # 48225*1

        if self.config['loss_type'] == 'CE':
            loss = self.criterion(outs, labels.long())
        else:
            loss = naive_loss(outs_forNavieLoss, labels_forNavieLoss, train_idx,
                              loss_weight=self.naive_loss_weight, ohem=self.naive_ohem, focal=self.naive_focal,
                              gamma=self.config['gamma'])

        train_loss += loss.item()
        self.writer.add_scalar('train_loss', loss, global_step=self.optim_steps)
        '''变回'''
        # softmax_outs = torch.softmax(outs, dim=1)
        # y_pred = torch.argmax(softmax_outs, dim=1)
        # y_pred = torch.transpose(torch.reshape(y_pred, (class_num, -1)), 0, 1).cpu().detach().numpy()
        y_pred = torch.transpose(torch.reshape(torch.argmax(temp, dim=1), (class_num, -1)), 0, 1).cpu().detach().numpy()
        right_label = lbls[train_idx].cpu().detach().numpy()
        # pred_probs = softmax_outs[:, 1].detach().cpu().numpy()
        # pred_probs = pred_probs.reshape(-1, class_num)
        F1Measure_value, Hamming_Loss_value, importantValue = evaluation(right_label, y_pred)
        # F1Measure_value, Hamming_Loss_value, importantValue = evaluation(right_label, pred_probs)

        '''反向传播'''
        # loss.requires_grad_(True)
        loss.backward()
        self.optim.step()
        self.optim_steps += 1

        # '''计算评价指标'''
        # with torch.no_grad():
        #     softmax_outs = torch.softmax(outs, dim=1)
        #     y_pred = torch.argmax(softmax_outs, dim=1)
        #     y_pred = torch.transpose(torch.reshape(y_pred, (class_num, -1)), 0, 1).cpu().detach().numpy()
        #     right_label = lbls[train_idx].cpu().detach().numpy()
        #     pred_probs = softmax_outs[:, 1].detach().cpu().numpy()
        #     pred_probs = pred_probs.reshape(-1, class_num)
        #     F1Measure_value, Hamming_Loss_value, importantValue = evaluation(right_label, pred_probs)

        temp = {'F1': F1Measure_value, 'y_pred': y_pred, 'Hamming': Hamming_Loss_value}
        if self.config['task'] == 'classification':
            return train_loss, temp, importantValue
        else:
            return train_loss, temp, importantValue

    def _valid_step(self, about_epoch, X, A, lbls, val_idx):
        self.net.eval()
        valid_loss = 0
        print(about_epoch)
        all_pred_probs = []
        all_true_labels = []

        '''计算'''
        with torch.no_grad():
            outs, embbeding_y = self.net(X, A)  # outs: list, 15*4019*2
            outs_forNavieLoss = outs  # 15*4019*2
            labels = lbls[val_idx]  # 4019*15 -> 402*15
            labels = labels.to(torch.float)
            labels_forNavieLoss = labels
            '''展开'''
            # 将outs转换为 6030*2
            temp = outs[0][np.asarray(val_idx)]  # temp: 402*2
            for i in range(class_num - 1):
                temp = torch.concat((temp, outs[i + 1][np.asarray(val_idx)]), dim=0)
            outs = temp  # 6030*2      6030 = 402*15
            # 将label转换为 6030*1
            labels = torch.reshape(torch.transpose(labels, 0, 1), (-1,))  # 6030*1

            if self.config['loss_type'] == 'CE':
                loss = self.criterion(outs, labels.long())
            else:
                loss = naive_loss(outs_forNavieLoss, labels_forNavieLoss, val_idx,
                                  loss_weight=self.naive_loss_weight, ohem=self.naive_ohem, focal=self.naive_focal,
                                  gamma=self.config['gamma'])

        valid_loss += loss.item()
        '''变回'''
        # softmax_outs = torch.softmax(outs, dim=1)
        # y_pred = torch.argmax(softmax_outs, dim=1)
        # y_pred = torch.transpose(torch.reshape(y_pred, (class_num, -1)), 0, 1).cpu().detach().numpy()
        y_pred = torch.transpose(torch.reshape(torch.argmax(temp, dim=1), (class_num, -1)), 0, 1).cpu().detach().numpy()
        right_label = lbls[val_idx].cpu().detach().numpy()
        # pred_probs = softmax_outs[:, 1].detach().cpu().numpy()
        # pred_probs = pred_probs.reshape(-1, class_num)
        # F1Measure_value, Hamming_Loss_value, importantValue = evaluation(right_label, pred_probs)
        F1Measure_value, Hamming_Loss_value, importantValue = evaluation(right_label, y_pred)

        """可视化"""
        visualize(about_epoch, embbeding_y, right_label, val_idx, config, self.draw_data)

        """auc需要预测概率"""
        pred_probs = F.softmax(outs, dim=1)[:, 1].cpu().numpy()
        pred_probs = pred_probs.reshape(-1, class_num)  # 重新形状以适应多类别

        # 记录真实标签和预测概率
        all_true_labels = right_label.reshape(-1, class_num)
        all_pred_probs = pred_probs

        temp = {'F1': F1Measure_value, 'y_pred': y_pred, 'Hamming': Hamming_Loss_value, 'true_probs': all_true_labels, 'pred_probs': all_pred_probs}
        if self.config['task'] == 'classification':
            return valid_loss, temp, importantValue
        else:
            return valid_loss, temp, importantValue

    def save_ckpt(self, epoch):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
            'best_metric': self.best_metric,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.log_dir, 'saved_model', 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.log_dir + '/saved_model/', 'checkpoint', 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth, map_location='cuda')
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        """准备"""
        write_record(self.txtfile, self.config)
        # self.net = self.net.to('cuda')

        self.net = self.net.to('cuda')
        # 使用 DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.net = nn.DataParallel(self.net)

        val_metric_list, test_metric_list = [], []
        x, t_loss, v_loss, te_loss = [], [], [], []  # loss 图
        train_loss_lines, val_loss_lines, te_loss_lines = None, None, None
        """数据"""
        num_vertices, peptideSeq, lbl, edge, self.train_mask, self.val_mask, self.test_mask, self.pep_seq_raw = self.data
        lbl = torch.Tensor(lbl)
        X = self._get_Embedding(peptideSeq)
        self.X = X
        G = Hypergraph(num_vertices, edge)
        if self.config['frontPart'] == 'textCNN':
            self.X = X.to(self.device)
        else:
            self.X = X.to(self.device)
        self.lbls = lbl.to(self.device)
        self.G = G.to(self.device)
        self.net = self.net.to(self.device)
        """训练"""
        all_true_labels = []
        all_pred_probs = []
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            print("Epoch {} cur_lr {}".format(i, self.optim.param_groups[1]['lr']))
            """train valid test 步骤"""
            train_loss, train_metric, _ = self._train_step(self.X, self.G, self.lbls, self.train_mask)
            valid_loss, val_metric, val_importantValue = self._valid_step(f'valid_epoch_{i}', self.X, self.G, self.lbls,
                                                                          self.val_mask)
            test_loss, test_metric, test_importantValue = self._valid_step(f'test_epoch_{i}', self.X, self.G, self.lbls,
                                                                           self.test_mask)

            # 最好的val再保存test
            if self.best_val_importantValue is None:
                self.best_val_importantValue = val_importantValue
                self.best_test_importantValue = test_importantValue
            if self.best_val_importantValue['average'][4] < val_importantValue['average'][4]:
                self.best_val_importantValue = val_importantValue
                self.best_test_importantValue = test_importantValue
                self.best_test_importantValue['epoch'] = i
                self.change_status = True

            """画loss图"""
            if (i % 50 == 0) or (i == 1):
                x, t_loss, v_loss, te_loss, train_loss_lines, val_loss_lines, te_loss_lines = \
                    visualize_LossGraph(config, self.log_dir, i, x,
                                        t_loss, v_loss, te_loss,
                                        train_loss, valid_loss, test_loss,
                                        train_loss_lines, val_loss_lines, te_loss_lines)

            """保存模型"""
            val_metric_list.append(val_metric)
            test_metric_list.append(test_metric)
            path = os.path.join(self.log_dir, 'saved_model')
            os.makedirs(path, exist_ok=True)
            if self.config['save_model'] == 'best_valid':
                if (self.config['task'] == 'regression' and (self.best_metric > val_metric)) or (
                        self.config['task'] == 'classification' and self.change_status):
                    self.best_metric = val_importantValue['average'][4]
                    self.change_status = False
                    torch.save(self.net.state_dict(),
                               os.path.join(self.log_dir + '/saved_model/', 'best_valid_model.pth'))
                    # 画AUC - ROC曲线
                    true_labels = val_metric['true_probs']
                    pred_probs = val_metric['pred_probs']
                    self.plot_auc_roc(true_labels, pred_probs, i)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=i)
            self.writer.add_scalar('test_loss', test_loss, global_step=i)

            if self.config['task'] == 'classification':
                print(f'train_loss:{train_loss} val_loss:{valid_loss} test_loss:{test_loss}\n'
                      f'train_F1Measure:{sum(train_metric["F1"])/len(train_metric["F1"])} val_F1Measure:{sum(val_metric["F1"])/len(val_metric["F1"])} test_F1Measure:{sum(test_metric["F1"])/len(test_metric["F1"])}')
            else:
                pass
            write_record(self.txtfile,
                         f'epoch:{i} train_loss:{train_loss} val_loss:{valid_loss} test_loss:{test_loss}\n'
                         f'train_metric:{train_metric} val_metric:{val_metric} test_metric:{test_metric}')
            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i)
            print("=" * 180 + '\n\n')
        print("最好AUC的val再保存test数据 -> \n", self.best_test_importantValue)

    def plot_auc_roc(self, true_labels, pred_probs, epoch):
        # print(true_labels)
        # print(pred_probs)
        # print(epoch)

        for class_idx in range(15):
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve([true[class_idx] for true in true_labels], [pred[class_idx] for pred in pred_probs])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_idx + 1} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Class {class_idx + 1}')
            plt.legend(loc="lower right")
            save_path = f'{self.log_dir}/visualization/auc_roc_{epoch}/roc_curve_class_{class_idx + 1}.svg'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()
