import pickle
import csv
import os
import random
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

"""
TAO 预处理

new task : 
0 seq文件读入
1 构建不同输入            3066raw
1.1 重复的边都去掉     1275left
1.2 重复的自环去掉     2239left
1.3 自环去掉                1315left

5614 对PR
4019 个P
3066 个R
489 个没有pid的seq
15 个类别

"""
# config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)
class_num = 15
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
data_names = ['Anal_canal_cancer', 'Bile_duct_cancer', 'Bladder_cancer', 'Breast_cancer',
              'Colon_cancer', 'Gastric_cancer', 'Kidney_cancer', 'Leukemia', 'Liver_cancer',
              'Lung_cancer', 'Ovary_cancer', 'Prostate_cancer', 'Skin_cancer', 'Thyroid_cancer', 'Tongue_cancer'
              ]
acid_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def read_file(path, cols):
    n_row = 0
    data = []
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            onedata = []
            maybe = row[0].split('\t')
            if len(maybe) > 2:
                row = maybe
            for col in cols:
                if '>' not in row[col]:  # ----------------------------------------------------
                    onedata.append(row[col])  # 选择某一列加入到data数组中
            if len(onedata) > 0:
                data.append(onedata)
            n_row = n_row + 1

    return data, n_row


def save_file(path, data):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def split_data(train_batch_size=401 * 8, test_batch_size=410, val_batch_size=401):
    # """" 现在去除离群点后还剩 3508 """
    """-7-11-13-14后还剩 4019"""
    # index_list = [1 for _ in range(2808)] + [2 for _ in range(350)] + [3 for _ in range(350)]
    index_list = [1 for _ in range(3215)] + [2 for _ in range(402)] + [3 for _ in range(402)]  # 8:1:1
    # index_list = [1 for _ in range(1340)] + [2 for _ in range(1340)] + [3 for _ in range(1339)]  # 1:1:1
    random.shuffle(index_list)
    train_data, test_data, val_data = [], [], []
    for i in index_list:
        if i == 1:
            train_data.append(True)
            test_data.append(False)
            val_data.append(False)
        if i == 2:
            train_data.append(False)
            test_data.append(True)
            val_data.append(False)
        if i == 3:
            train_data.append(False)
            test_data.append(False)
            val_data.append(True)
    return train_data, test_data, val_data


def create_peptide_dict(Pid_Pseq_Label):
    # 字典 {Pid : Pseq}
    psDict = {}
    # 字典 {Pid : Label}
    plDict = {}
    for row in Pid_Pseq_Label:
        psDict[row[0]] = row[1]
        plDict[row[0]] = row[2]
    spDict = {v: k for k, v in psDict.items()}
    return psDict, plDict, spDict


def create_RNA_dict(Pseq_Rid, spDict):
    # 字典 {Rid : 频率}
    rDict = {}
    # 字典 {Pid : [Rid...]}
    cDict = {}
    # 没有id的Pseq ???删除
    errorData = []
    for pair in Pseq_Rid:
        Rids = pair[1].split(',')
        # 构建cDict
        if pair[0] in spDict.keys():  # 如果是好数据
            pid = spDict[pair[0]]
            if pid in cDict.keys():  # 如果pid在cDict里
                for i in Rids:  # 合并
                    if i not in cDict[pid]:
                        cDict[pid].append(i)
            else:  # 新建
                cDict[pid] = Rids
            # 构建rDict (好数据才弄)
            for rid in Rids:
                if rid in rDict.keys():
                    rDict[rid] = rDict[rid] + 1
                else:
                    rDict[rid] = 1
        else:  # 坏数据
            errorData.append(pair[0])
    return rDict, cDict, errorData


def create_edges(rDict, cDict):
    # 遍历, 对每个Rid构建超边, [[Pid...]...]     (nR行?列)
    hyper_edges = []
    for rid in rDict.keys():
        one_hyper_edge = []
        for pid, rids in cDict.items():
            if rid in rids:
                one_hyper_edge.append(int(pid))
        one_hyper_edge.sort()
        if len(one_hyper_edge) == 0:
            print("wrong rid ???", rid)
        hyper_edges.append(one_hyper_edge)
    return hyper_edges


def create_labels(plDict):
    # 二维list [[label...]...]        (nP行?列)
    hyper_labels = []
    lDict = {}
    hyper_labels2 = []
    for pid, label in plDict.items():
        # hyper_labels.append(label.split('/'))
        hyper_labels.append(label.split('/'))
        for l in label.split('/'):
            if l in lDict.keys():
                lDict[l] = lDict[l] + 1
            else:
                lDict[l] = 1
    llist = list(lDict.keys())
    if '' in llist:
        llist.pop(llist.index(''))
    llist = sorted(llist)
    for row in hyper_labels:
        # one = [0 for _ in range(15)]
        one = [0 for _ in range(class_num)]  #
        for i in row:
            if i != '':
                one[llist.index(i)] += 1
        hyper_labels2.append(one)

    return hyper_labels2, lDict


def create_pseq(psDict):
    # Pseq 保存为 ['OIUDS',...]
    pid_pseq = []
    for pid in psDict.keys():
        pid_pseq.append(psDict[pid])
    return pid_pseq


def edge_remove_repeated(hyper_edges):
    edges = []
    for e in hyper_edges:
        ifrepeated = 0
        for other in hyper_edges:
            if other == e:
                ifrepeated += 1
        if ifrepeated == 1:
            edges.append(e)
    return edges


def edge_remove_repeated_selfloop(hyper_edges):
    edges = []
    selfloop = []
    for e in hyper_edges:
        ifrepeated = False
        for other in hyper_edges:
            if other == e and len(e) == 1:
                ifrepeated = True
                if e not in selfloop:
                    selfloop.append(e)
        if not ifrepeated:
            edges.append(e)
    edges += selfloop
    return edges


def edge_remove_selfloop(hyper_edges):
    edges = []
    for e in hyper_edges:
        if len(e) > 1:
            edges.append(e)
    return edges


def generate_frequency_matrix(hyper_edges, hyper_labels):
    freMat = np.zeros((15, 15), dtype=np.int)
    for e in hyper_edges:
        for v1 in e:
            for v2 in e:
                if v1 != v2:
                    oneEdgeFre = np.sum([hyper_labels[v1], hyper_labels[v2]], axis=0).tolist()
                    for i, f in enumerate(oneEdgeFre):
                        if f > 0:
                            for j, f2 in enumerate(oneEdgeFre):
                                if j >= i:
                                    break
                                if f2 > 0:
                                    freMat[i][j] += 1
    print(freMat)
    return freMat


def generate_orf_edgeAndlabel(Plongids, labels, Orfid, Plongid_list):
    hyper_edges = []
    hyper_labels = []

    label_dict_list = []
    for ls in labels:
        for l in ls:
            if l not in label_dict_list:
                label_dict_list.append(l)
    label_dict_list.sort()
    print(label_dict_list[:10])

    plonglDict = {}
    for i, Plongid in enumerate(Plongids):
        plonglDict[Plongid] = [0 for _ in range(class_num)]
        for l in labels[i]:
            plonglDict[Plongid][label_dict_list.index(l)] += 1
    for a, b in plonglDict.items():
        hyper_labels.append(b)
    # print(plonglDict)
    print(hyper_labels[:10])

    for i, orfid in enumerate(Orfid):
        one_hyper_edge = Plongid_list[i]
        # one_hyper_edge.sort()
        hyper_edges.append(one_hyper_edge)

    freMat = np.zeros((class_num, class_num))
    plongid = list(plonglDict.keys())
    num_cancer = [0 for i in range(class_num)]
    for idx, e in enumerate(tqdm(hyper_edges)):
        oneEdgeFre = [0 for _ in range(class_num)]
        for v in e:
            if v == 'SPENP00':
                print('idx', idx)
                break
            oneEdgeFre = list(np.sum([oneEdgeFre, hyper_labels[plongid.index(v)]], axis=0))

        for i, f in enumerate(oneEdgeFre):
            # if f > 0:
            if f > 1:
                num_cancer[i] += f
        for i, f in enumerate(oneEdgeFre):
            # if f > 0:
            if f > 1:
                for j, f2 in enumerate(oneEdgeFre):
                    if j >= i:
                        break
                    if f2 > 0:
                        # freMat[i][j] += oneEdgeFre[i] * oneEdgeFre[j]
                        freMat[i][j] += 1

    print(freMat)
    # 除最大减最小
    maxF, minF = 0, 10000
    [rows, cols] = freMat.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            if freMat[i, j] > maxF:
                maxF = freMat[i, j]
            if freMat[i, j] < minF and freMat[i, j] > 0:
                minF = freMat[i, j]
    print('maxF', maxF, 'minF', minF)
    for i in range(rows):
        for j in range(cols):
            # freMat[i, j] = (freMat[i, j] - minF) / (maxF - minF)
            freMat[i, j] = freMat[i, j]
    # for i in range(rows):
    #     for j in range(cols):
    #         freMat[i, j] = freMat[i, j] / (num_cancer[i] * num_cancer[j])
    print(freMat)
    print(num_cancer)
    return freMat


def plot(freMat, name):
    lendata = len(data_names)
    p_value = np.zeros((lendata, lendata))
    pearson_r = np.zeros((lendata, lendata))

    candidate = []

    for i in range(lendata):
        for j in range(lendata):
            p_value[i, j] = freMat[i][j]
            # pearson_r[i, j] = (r - 0.99) * 100
            # pearson_r[i, j] = r

    print(pearson_r)
    print(p_value)

    mask = np.zeros_like(p_value)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        plt.rcParams['savefig.dpi'] = 600  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        f, ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(left=0.2, bottom=0.18, right=0.95, top=0.9, hspace=0.25, wspace=0.27)
        ax = sns.heatmap(p_value, xticklabels=data_names, yticklabels=data_names,
                         # vmax = 1,
                         # vmin = 0,
                         mask=mask, linewidths=.5, cmap="YlGnBu")
        plt.title(name)
        # plt.savefig('all.svg')
        plt.show()


def remove_outlier(psDict, plDict, spDict, rDict, cDict):
    # 字典 {Pid : Pseq}       4019
    # 字典 {Pid : Label}      4019
    # 字典 {Rid : 频率}         3066
    # 字典 {Pid : [Rid...]}       4019
    outlier_pid_list = []
    # 选出离群点  511个
    for oneP in cDict.keys():
        if len(cDict[oneP]) == 1:
            if rDict[cDict[oneP][0]] == 1:
                outlier_pid_list.append(oneP)
    print(outlier_pid_list)
    # 删除离群点
    temp = cDict
    for outone in outlier_pid_list:
        psDict.pop(outone)
        plDict.pop(outone)
        cDict.pop(outone)
        # rDict 好麻烦
        if outone in cDict.keys():  # 为啥有的pid没有关系
            for rid in cDict[outone]:
                rDict[rid] -= 1
                if rDict[rid] == 0:
                    rDict.pop(rid)
    spDict = {v: k for k, v in psDict.items()}
    print()
    # 字典 {Pid : Pseq}       3508 = 4019 - 511
    # 字典 {Pid : Label}      3508
    # 字典 {Rid : 频率}         3066
    # 字典 {Pid : [Rid...]}       3508
    return [psDict, plDict, spDict, rDict, cDict]


def remove_class(psDict, plDict, spDict, rDict, cDict, NO):
    # 字典 {Pid : Pseq}       4019
    # 字典 {Pid : Label}      4019
    # 字典 {Rid : 频率}         3066
    # 字典 {Pid : [Rid...]}       4019
    bad_pid = []
    for p, l in plDict.items():
        if NO in l:
            plDict[p] = plDict[p].replace('/' + NO, '')
            plDict[p] = plDict[p].replace(NO + '/', '')
            plDict[p] = plDict[p].replace(NO, '')
            if plDict[p] == '':
                bad_pid.append(p)

    # 字典 {Pid : Pseq}       3508 = 4019 - 511
    # 字典 {Pid : Label}      3508
    # 字典 {Rid : 频率}         3066
    # 字典 {Pid : [Rid...]}       3508
    return [psDict, plDict, spDict, rDict, cDict]


if __name__ == '__main__':
    # Pseq_Rid, n_con = read_file('./raw/Pseq_Rid.csv', [0, 1])
    # Pid_Pseq_Label, n_P = read_file('./raw/Pid_Pseq_Label.csv', [0, 1, 2])
    #
    # # psDict, plDict, spDict = create_peptide_dict(Pid_Pseq_Label)
    # # rDict, cDict, errorData = create_RNA_dict(Pseq_Rid, spDict)
    # # hyper_edges = create_edges(rDict, cDict)
    # # hyper_labels, lDict = create_labels(plDict)
    # # input_Pseq = create_pseq(psDict)
    # psDict, plDict, spDict = create_peptide_dict(Pid_Pseq_Label)
    # rDict, cDict, errorData = create_RNA_dict(Pseq_Rid, spDict)

    # # hyper_edges_remove_repeated = edge_remove_repeated(hyper_edges)
    # # hyper_edges_remove_repeated_selfloop = edge_remove_repeated_selfloop(hyper_edges)
    # # hyper_edges_remove_selfloop = edge_remove_selfloop(hyper_edges)
    # # 以上 yuan 3066   1275  2239  1315

    # # 生成mask文件
    # train_data, test_data, val_data = split_data()
    # save_file('./process/train_mask_811.pkl', train_data)
    # save_file('./process/test_mask_811.pkl', test_data)
    # save_file('./process/val_mask_811.pkl', val_data)
    # print('save mask finish...')
    #
    # """  保存文件  """
    #
    # # save_file('../dataset/process/peptide_seq.pkl', input_Pseq)
    # # save_file('../dataset/process/hyper_edges.pkl', hyper_edges)
    # # save_file('../dataset/process/hyper_labels.pkl', hyper_labels)
    # save_file('./process/hyper_edges_-7-11-13-14.pkl', hyper_edges)
    # save_file('./process/hyper_labels_-7-11-13-14.pkl', hyper_labels)
    # save_file('./process/hyper_edges_new15.pkl', hyper_edges)
    # save_file('./process/hyper_labels_new15.pkl', hyper_labels)

    # save_file('./process/hyper_edges_new10.pkl', hyper_edges)
    # save_file('./process/hyper_labels_new10.pkl', hyper_labels)
    # print('save dataset finish...')