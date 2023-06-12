import os

import numpy as np
import torch
import yaml
from dhg import Hypergraph
from dhg.visualization import draw_hypergraph
from matplotlib import pyplot as plt

from src.Train_class import Trainer
from model.HGCPep_Model import HGCPep_Model
from utils.util_data import readDatasetFromPickle
from visualization.dhg_visualization.test_feature import test_draw_in_euclidean_space, test_draw_poincare_ball, test_draw_in_upsetplot
from visualization.dhg_visualization.test_visualize import test_vis_hypergraph

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def naive_loss(y_pred, y_true):
    # 用nn库函数计算loss
    loss_Fun = torch.nn.CrossEntropyLoss()
    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]

    loss_output = torch.zeros(num_examples).cuda()
    for i in range(num_task):
        loss_temp = loss_Fun(y_pred[i], y_true[:, i].long())
        loss_output += loss_temp
    loss = torch.sum(loss_output)

    return loss


def deal_edge(edge, threshold_value, lbl):
    edge_new = []
    vDcit = {}
    for i in range(len(edge)):
        one_edge = []
        for j in range(len(edge[i])):
            one_edge.append(edge[i][j])
            if edge[i][j] not in vDcit.keys():
                vDcit[edge[i][j]] = 1
            else:
                vDcit[edge[i][j]] += 1
        if len(one_edge) > 3:
            edge_new.append(one_edge)

    # 选出vDcit中键值最大的前 threshold_value 个节点
    max_vDcit = sorted(vDcit.items(), key=lambda item: item[1], reverse=True)[:threshold_value]
    # 对应节点的label
    lbl = {i: lbl[i] for i in vDcit.keys()}
    # 去除edge_new中编号不在 max_vDcit 中的节点
    edge_new = [i for i in edge_new if len(set(i).intersection(set([j[0] for j in max_vDcit]))) == len(i)]
    # 将edge_new中的节点编号重新映射
    label_new = {}
    for i in range(len(edge_new)):
        for j in range(len(edge_new[i])):
            for k in range(len(max_vDcit)):
                if edge_new[i][j] == max_vDcit[k][0]:
                    edge_new[i][j] = k
                    break
    return edge_new, lbl.values()


def draw_one_hypergraph(threshold_value):
    config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)

    num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask = readDatasetFromPickle(config)

    """处理edge"""
    lbl = np.argmax(lbl, axis=1)
    edge, lbl = deal_edge(edge, threshold_value, lbl)
    G = Hypergraph(threshold_value, edge)
    # color_list = plt.cm.tab10(np.linspace(0, 1, 12))
    # color_list = plt.cm.rainbow(np.linspace(0, 1, 15))
    # color_list = plt.get_cmap('rainbow')(range(15))

    # color_list = ['#FFE4C4','#668B8B','#FFE4E1','#528B8B','#BEBEBE','#7B68EE','#B4EEB4','#FFF68F','#8B814C','#CD5C5C','#FFA500','#FF1493','#8B0000','#C1CDC1','#0000CD','#87CEFF','#993366','#009999','#CC3399']
    # draw_hypergraph(G, e_style="circle", v_line_width=10, v_label=[str(i) for i in lbl], v_color=[color_list[i] for i in lbl], e_fill_color='#CCFFFF')
    # plt.savefig(f"./hyper_graph_{threshold_value}.pdf")
    # plt.show()


def test_model(model_path):
    config = yaml.load(open("./config_base_test.yaml", "r"), Loader=yaml.FullLoader)

    # 加载模型
    model = HGCPep_Model(config)
    # model = torch.load(model_path, map_location=torch.device('cuda:1'))
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")
    # 设置模型为评估模式
    model.eval()

    # 加载测试数据
    # trainer = Trainer(config, "../src/config_base.yaml")
    num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask = readDatasetFromPickle(config)
    lbl = torch.Tensor(lbl)
    G = Hypergraph(num_vertices, edge)
    X = peptideSeq.to("cuda")
    lbls = lbl.to("cuda")
    G = G.to("cuda")
    # net = trainer.net.to("cuda")

    # 定义损失函数和优化器
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 测试模型
    test_loss = 0.0
    test_acc = 0.0
    total = 0

    with torch.no_grad():
        # 将输入数据传入模型进行前向传播
        outputs, embbeding_y = model(X, G)

        """用 DHG 可视化"""
        path_base = './visualMAP_new10new/'
        if not os.path.exists(path_base):
            os.makedirs(path_base)
        # y_true = np.argmax(lbls.cpu(), axis=1)
        lbls = lbls.cpu().numpy()

        # 统计每个类别的个数
        many_class = {}
        many_class_index2label = {}
        y_true = []
        for i in range(len(lbls)):
            # 将lbls[i]变成字符串
            str_label = str(lbls[i])
            y_true.append(str_label)
            if str_label not in many_class.keys():
                many_class[str_label] = 1
                many_class_index2label[str_label] = [i]
            else:
                many_class[str_label] += 1
                many_class_index2label[str_label].append(i)
        # print(len(many_class.keys()))
        # print(many_class)
        # 将many_class中value小于50的类别删除
        many_class = {k: v for k, v in many_class.items() if
                      (v >= 50) and (k != '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]') and (
                                  k != '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]')}
        # 将many_class_index2label中key不在many_class中的删除
        many_class_index2label = {k: v for k, v in many_class_index2label.items() if k in many_class.keys()}
        # test_draw_in_euclidean_space(embbeding_y.cpu(), y_true, path_base, many_class_index2label)
        # test_draw_in_upsetplot(embbeding_y.cpu(), y_true, path_base, many_class_index2label)

        # 计算损失函数
        loss = naive_loss(outputs, lbls)

        # 统计测试集的损失和准确率
        test_loss += loss.item() * X.size(0)
        # _, predicted = torch.max(outputs, 1)
        total += lbls.size(0)
        # test_acc += (predicted == lbls).sum().item()

    # 打印测试结果
    print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(test_loss / total, test_acc / total))


if __name__ == '__main__':
    # # TODO: 指定你的pth文件的路径
    # name = ''
    model_path = f'./best_valid_model_10.pth'
    # model_path = f'./best_valid_model_15.pth'
    test_model(model_path)
    # draw_one_hypergraph(90)
