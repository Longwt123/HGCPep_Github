import pickle


labels10 = ['Anal canal cancer ncPEPs',  'Bladder cancer ncPEPs', 'Breast cancer ncPEPs',
            'Colon cancer ncPEPs',
            'Kidney cancer ncPEPs', 'Leukemia ncPEPs',
            'Lung cancer ncPEPs',
             'Prostate cancer ncPEPs', 'Skin cancer ncPEPs',
            'Tongue cancer ncPEPs']
labels15 = ['Anal canal cancer ncPEPs', 'Bile duct cancer ncPEPs', 'Bladder cancer ncPEPs', 'Breast cancer ncPEPs',
            'Colon cancer ncPEPs',
            'Gastric cancer ncPEPs', 'Kidney cancer ncPEPs', 'Leukemia ncPEPs', 'Liver cancer ncPEPs',
            'Lung cancer ncPEPs',
            'Ovary cancer ncPEPs', 'Prostate cancer ncPEPs', 'Skin cancer ncPEPs', 'Thyroid cancer ncPEPs',
            'Tongue cancer ncPEPs']

def readDatasetFromPickle(which_dataset):
    path = "../dataset/process/"
    if which_dataset == 15:
        dataset = {'label_file_name': 'hyper_labels_new15', 'edges_file_name': 'hyper_edges_new15',
                   'peptide_file_name': 'peptide_seq'}
    else:
        dataset = {'label_file_name': 'hyper_labels_new10', 'edges_file_name': 'hyper_edges_new10',
                   'peptide_file_name': 'peptide_seq'}
    num_vertices = 4019
    lbl = pickle.load(open(path + dataset['label_file_name'] + ".pkl", "rb"))
    peptideSeq = pickle.load(open(path + dataset['peptide_file_name'] + ".pkl", "rb"))
    edge = pickle.load(open(path + dataset['edges_file_name'] + ".pkl", "rb"))

    train_mask = pickle.load(open(path + 'train_mask_811' + ".pkl", "rb"))
    val_mask = pickle.load(open(path + 'val_mask_811' + ".pkl", "rb"))
    test_mask = pickle.load(open(path + 'test_mask_811' + ".pkl", "rb"))

    # peptideSeq = codePeptide(peptideSeq, config)

    """TAO 统计数据"""
    # tao_seeseeDataset(num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask)
    return num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask

def count_rows_with_value(data, column_index):
    count = 0
    for row in data:
        if row[column_index] == 1:
            count += 1
    return count


# 加载测试数据
# num_vertices, peptideSeq, lbl, edge, train_mask, val_mask, test_mask\
data_10 = readDatasetFromPickle(10)
label_10 = data_10[2]
data_15 = readDatasetFromPickle(15)
label_15 = data_15[2]

# print('11   Lung cancer ncPEPs ',count_rows_with_value(label_11, 8))
# print('15   Lung cancer ncPEPs ',count_rows_with_value(label_15, 9))
for i in range(10):
    print(label_15[i])
    print(label_10[i])
temp = []
for i in range(10):
    temp.append(count_rows_with_value(label_10, i))
print(f'10 的 ',temp)
temp = []
for i in range(15):
    temp.append(count_rows_with_value(label_15, i))
print(f'15 的 ',temp)

all_1 = []
for l in label_15:
    isgood = True
    for i in [0,2,3,4,6,7,9,11,12,14]:
        if l[i] != 1:
           isgood = False
    if isgood:
        all_1.append(label_15.index(l))
print(len(all_1))
print(all_1)