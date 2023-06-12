import yaml
from numpy import mean
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, \
    hamming_loss
import numpy as np

"""
HGCPep 评测
"""
config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)
class_num = config['class_num']


def evaluation(y_true, y_pred):

    every_acc = []
    every_precision = []
    every_recall = []
    every_MCC = []
    every_AUC = []
    F1Measure_value = []
    Hamming_Loss_value = []
    y_pred_probs = np.array(y_pred)

    for i in range(class_num):
        one_y_true = y_true[:, i]
        one_y_pred_prob = y_pred_probs[:, i]

        # 将概率转化为离散标签
        one_y_pred = (one_y_pred_prob >= 0.5).astype(int)

        every_acc.append(accuracy_score(one_y_true, one_y_pred))
        every_precision.append(precision_score(one_y_true, one_y_pred))
        every_recall.append(recall_score(one_y_true, one_y_pred))
        every_MCC.append(matthews_corrcoef(one_y_true, one_y_pred))
        every_AUC.append(roc_auc_score(one_y_true, one_y_pred_prob))  # 使用概率计算AUC
        F1Measure_value.append(f1_score(one_y_true, one_y_pred))
        Hamming_Loss_value.append(hamming_loss(one_y_true, one_y_pred))


    # 打印每个类别的性能
    print(',\t\t'.join([str(round(i, 4)) for i in every_acc]))
    print(',\t\t'.join([str(round(i, 4)) for i in every_precision]))
    print(',\t\t'.join([str(round(i, 4)) for i in every_recall]))
    print(',\t\t'.join([str(round(i, 4)) for i in every_MCC]))
    print(',\t\t'.join([str(round(i, 4)) for i in every_AUC]))

    importantValue = {'acc': every_acc,
                      'pre': every_precision,
                      'recall': every_recall,
                      'mcc': every_MCC,
                      'auc': every_AUC,
                      'average': [mean(every_acc),
                                  mean(every_precision),
                                  mean(every_recall),
                                  mean(every_MCC),
                                  mean(every_AUC)]
                      }

    print('F1Measure_mean: ', mean(F1Measure_value),'Hamming_Loss_mean: ', mean(Hamming_Loss_value))
    print('-' * 100)

    return F1Measure_value, Hamming_Loss_value, importantValue
