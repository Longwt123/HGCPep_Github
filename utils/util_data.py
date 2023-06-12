import pickle
import re
import torch
import torch.nn.utils.rnn as rnn_utils
import yaml
import random


aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
           'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
           'W': 20, 'Y': 21, 'V': 22, 'X': 23}


def readDatasetFromPickle(config):
    path = "../dataset/process/"
    dataset = config['dataset']
    num_vertices = 4019
    lbl = pickle.load(open(path + dataset['label_file_name'] + ".pkl", "rb"))
    peptideSeq = pickle.load(open(path + dataset['peptide_file_name'] + ".pkl", "rb"))
    edge = pickle.load(open(path + dataset['edges_file_name'] + ".pkl", "rb"))

    train_mask = pickle.load(open(path + dataset['train_mask_file_name'] + ".pkl", "rb"))
    val_mask = pickle.load(open(path + dataset['valid_mask_file_name'] + ".pkl", "rb"))
    test_mask = pickle.load(open(path + dataset['test_mask_file_name'] + ".pkl", "rb"))

    # print(len(peptideSeq), len(peptideSeq[0]), len(peptideSeq[2]), len(peptideSeq[4]))
    # print(peptideSeq[:3])
    peptideSeq_emb = codePeptide(peptideSeq, config)
    # print(len(peptideSeq), len(peptideSeq[0]), len(peptideSeq[2]), len(peptideSeq[4]))
    # print(peptideSeq[:3])

    """TAO 统计数据"""
    # tao_seeseeDataset(num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask)

    """TAO 超图 -> 普通图"""
    # print(edge[0:5])
    # print(len(edge))

    # edge = simplify_edges(edge)

    # print(edge[0:5])
    # print(len(edge))


    return num_vertices, peptideSeq_emb, lbl, edge, train_mask, val_mask, test_mask, peptideSeq



import random
from itertools import combinations


def simplify_edges(edges):
    """
    将超图的边简化为普通图的边。

    参数：
        edges (list of lists): 超图的边列表，每个子列表表示一条边，包含多个节点。

    返回：
        simplified_edges (list of lists): 简化后的边列表，每条边只包含两个节点。
    """
    simplified_edges = []

    for edge in edges:
        num_nodes = len(edge)

        if num_nodes <= 2:
            simplified_edges.append(edge)

        elif 3 <= num_nodes <= 20:
            # 全连接成边
            for combo in combinations(edge, 2):
                simplified_edges.append(list(combo))

        elif 21 <= num_nodes <= 100:
            # 全连接成边，随机取20%的边
            all_edges = list(combinations(edge, 2))
            sample_size = int(0.2 * len(all_edges))
            sampled_edges = random.sample(all_edges, sample_size)
            for combo in sampled_edges:
                simplified_edges.append(list(combo))

        elif 101 <= num_nodes <= 200:
            # 全连接成边，随机取5%的边
            all_edges = list(combinations(edge, 2))
            sample_size = int(0.05 * len(all_edges))
            sampled_edges = random.sample(all_edges, sample_size)
            for combo in sampled_edges:
                simplified_edges.append(list(combo))

        else:
            # 全连接成边，随机取0.1%的边
            all_edges = list(combinations(edge, 2))
            sample_size = int(0.001 * len(all_edges))
            sampled_edges = random.sample(all_edges, sample_size)
            for combo in sampled_edges:
                simplified_edges.append(list(combo))

    return simplified_edges



# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
#
# def tao_seeseeDataset(num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask):
#     # 输出基础信息
#     print("num_vertices:", num_vertices)
#     print("数据类型: peptideSeq:", type(peptideSeq), ", lbl:", type(lbl), ", edge:", type(edge), ", train_mask:",
#           type(train_mask))
#     print("数据长度: peptideSeq:", len(peptideSeq), ", lbl:", len(lbl), ", edge:", len(edge), ", train_mask:",
#           len(train_mask))
#
#     # 初始化统计变量
#     num = [0, 0]  # 统计超边数量
#     all_label = [0 for _ in range(15)]  # 统计每个标签的数量
#     edge_lengths = []  # 统计每条边的节点数
#
#     for e in edge:
#         edge_lengths.append(len(set(e)))  # 统计去重后每条边的节点数
#         if len(set(e)) > 1:  # 超边去重后长度大于1
#             num[1] += 1
#         for v in e:
#             l = lbl[v]
#             all_label = [i + j for i, j in zip(all_label, l)]
#
#     # 输出统计结果
#     print("超边统计结果:", num)
#     print("标签统计结果:", all_label)
#
#     # 将标签统计结果标准化并输出
#     normalized_labels = [round(i / num_vertices, 3) for i in all_label]
#     print("标准化标签统计结果:", normalized_labels)
#
#     # 创建数据表格
#     label_data = {
#         'Label': [f'Label_{i}' for i in range(15)],
#         'Count': all_label,
#         'Normalized': normalized_labels
#     }
#     label_df = pd.DataFrame(label_data)
#     print(label_df)
#
#     # 统计每条边的节点数分布并排序
#     edge_length_counts = Counter(edge_lengths)
#     edge_length_data = {
#         'Edge Length': list(edge_length_counts.keys()),
#         'Count': list(edge_length_counts.values())
#     }
#     edge_length_df = pd.DataFrame(edge_length_data).sort_values(by='Edge Length')
#     print(edge_length_df)
#
#     # 绘制标签分布图表
#     plt.figure(figsize=(10, 6))
#     plt.bar(label_df['Label'], label_df['Count'], color='blue', alpha=0.7, label='Count')
#     plt.xlabel('Label')
#     plt.ylabel('Count')
#     plt.title('Label Distribution')
#     plt.legend()
#     plt.savefig('/mnt/sdb/home/lwt/tao/HGCPep_new/utils/statistics/fig0.png')
#     plt.show()
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(label_df['Label'], label_df['Normalized'], color='green', alpha=0.7, label='Normalized')
#     plt.xlabel('Label')
#     plt.ylabel('Normalized Count')
#     plt.title('Normalized Label Distribution')
#     plt.legend()
#     plt.savefig('/mnt/sdb/home/lwt/tao/HGCPep_new/utils/statistics/fig1.png')
#     plt.show()
#
#     # 绘制边长度分布图表
#     plt.figure(figsize=(10, 6))
#     plt.bar(edge_length_df['Edge Length'], edge_length_df['Count'], color='orange', alpha=0.7,
#             label='Edge Length Count')
#     plt.xlabel('Edge Length')
#     plt.ylabel('Count')
#     plt.title('Edge Length Distribution')
#     plt.legend()
#     plt.savefig('/mnt/sdb/home/lwt/tao/HGCPep_new/utils/statistics/fig2.png')
#     plt.show()
#
#     return label_df, edge_length_df





def codePeptide(peptideSeq, config):
    pep_seq = []
    pep_codes = []
    max_seq_len = 70

    for pep in peptideSeq:
        input_seq = ' '.join(pep)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)

        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))

    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)

    if config['frontPart'] == 'textCNN':
        return data
    if config['frontPart'] == 'prot_bert_bfd':
        return pep_seq
    return data


if __name__ == '__main__':
    path = "../dataset/process/"
    edge = pickle.load(open(path + "hyper_edges_remove_selfloop" + ".pkl", "rb"))
    point = []
    for i in edge:
        for j in i:
            point.append(j)
    point = set(point)
    print(len(point))

    # a = [ 0 for i in range(300)]
    #
    # for i in edge:
    #     a[len(i)] += 1
    # print(a)
