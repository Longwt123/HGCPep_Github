import numpy as np
import os

import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import umap
from sklearn.preprocessing import StandardScaler


def draw_umap_plot(about_epoch, repres_list, label_list, draw_data, n=2):
    # draw_umap_plot(about_epoch, embbeding_x, y_true, draw_data, 2)
    plt.rcParams['savefig.dpi'] = 800  # 图片像素
    plt.rcParams['figure.dpi'] = 800  # 分辨率
    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    cmap = ListedColormap(['#00beca', '#f87671'])
    if draw_data['draw_class'] == 'all':
        # 将onehot转化为数字
        label_list = np.argmax(label_list, axis=1)
        # # 挑出只有一个类别的数据
        # only_1class_index = []
        # for i in range(len(label_list)):
        #     if sum(label_list[i]) == 1:
        #         only_1class_index.append(i)
    else:
        # 将onehot转化为正负例
        new_data_index = []
        for i in range(len(label_list)):
            if label_list[i][draw_data['draw_class']] == 1:
                new_data_index.append(1)
            else:
                new_data_index.append(0)
        label_list = new_data_index

    repres = np.array(repres_list.cpu().detach().numpy())
    label = np.array(label_list)
    scaled_data = StandardScaler().fit_transform(repres)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_data)
    colors = np.array(["#00beca", "#f87671"])
    # print(embedding)
    fig, ax = plt.subplots()

    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=label, cmap='rainbow', s=5
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.gca().set_aspect('equal', 'datalim')

    if not os.path.exists(draw_data['dirs'] + 'HighQuality/'):
        os.makedirs(draw_data['dirs'] + 'HighQuality/')
    plt.savefig(
        draw_data['dirs'] + 'HighQuality/可视化uMap_{}.svg'.format(draw_data['withorwithoutHG'] + '_' + about_epoch))
    plt.savefig(
        draw_data['dirs'] + 'HighQuality/可视化uMap_{}.pdf'.format(draw_data['withorwithoutHG'] + '_' + about_epoch))
    plt.savefig(draw_data['dirs'] + '可视化uMap_{}.png'.format(draw_data['withorwithoutHG'] + '_' + about_epoch))
    plt.show()
    f = plt.gcf()  # 获取当前图像
    # f.savefig()
    f.clear()  # 释放内存


def visualize_PCA(epoch, data, data_index, draw_data, class_num):
    #  visualize_PCA(about_epoch, embbeding_x, y_true, 'without HyperGraph', 2)
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    data = data.cpu().detach().numpy()
    cmap = ListedColormap(['#00beca', '#f87671'])

    # # 挑出只有一个类别的数据
    # only_1class_index = []
    # for i in range(len(data_index)):
    #     if sum(data_index[i]) == 1:
    #         only_1class_index.append(i)

    X_pca = PCA(n_components=2).fit_transform(data)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure()
    new_data_index = []
    if draw_data['draw_class'] == 'all':
        # 将onehot转化为数字
        new_data_index = np.argmax(data_index, axis=1)
    else:
        # 将onehot转化为正负例
        for i in range(len(data_index)):
            if data_index[i][draw_data['draw_class']] == 1:
                new_data_index.append(1)
            else:
                new_data_index.append(0)
        new_data_index = np.array(new_data_index)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=new_data_index, alpha=0.6, cmap='rainbow')  # s：点的大小  c：点的颜色
    plt.title(f'PCA  epoch{epoch} {draw_data["withorwithoutHG"]}', fontdict=font)

    # if data_label is None:
    cbar = plt.colorbar(ticks=range(class_num))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(0 - 0.5, class_num - 0.5)
    if not os.path.exists(draw_data["dirs"] + 'HighQuality/'):
        os.makedirs(draw_data["dirs"] + 'HighQuality/')
    plt.savefig(draw_data['dirs'] + 'HighQuality/可视化PCA_{}.svg'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.savefig(draw_data['dirs'] + 'HighQuality/可视化PCA_{}.pdf'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.savefig(draw_data['dirs'] + '可视化PCA_{}.png'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.show()
    f = plt.gcf()  # 获取当前图像
    # f.savefig()
    f.clear()  # 释放内存


def visualize_t_SNE(epoch, data, data_index, draw_data, class_num):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    data = data.cpu().detach().numpy()
    cmap = ListedColormap(['#C6E5A9', '#B795D9'])

    # # 挑出只有一个类别的数据
    # only_1class_index = []
    # for i in range(len(data_index)):
    #     if sum(data_index[i]) == 1:
    #         only_1class_index.append(i)

    X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure()
    new_data_index = []
    if draw_data['draw_class'] == 'all':
        # 将onehot转化为数字
        new_data_index = np.argmax(data_index, axis=1)
    else:
        # 将onehot转化为正负例
        for i in range(len(data_index)):
            if data_index[i][draw_data['draw_class']] == 1:
                new_data_index.append(1)
            else:
                new_data_index.append(0)

        new_data_index = np.array(new_data_index)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=new_data_index, alpha=0.6, cmap='rainbow')
    # if data_label:
    #     for i in range(len(X_tsne)):
    #         plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
    #                      xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
    plt.title(f't-SNE  epoch{epoch} {draw_data["withorwithoutHG"]}', fontdict=font)

    cbar = plt.colorbar(ticks=range(class_num))
    # cbar.set_label(label='digit value', fontdict=font)
    plt.clim(0 - 0.5, class_num - 0.5)

    if not os.path.exists(draw_data["dirs"] + 'HighQuality/'):
        os.makedirs(draw_data["dirs"] + 'HighQuality/')
    plt.savefig(draw_data['dirs'] + 'HighQuality/可视化tSNE_{}.svg'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.savefig(draw_data['dirs'] + 'HighQuality/可视化tSNE_{}.pdf'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.savefig(draw_data['dirs'] + '可视化tSNE_{}.png'.format(draw_data['withorwithoutHG'] + '_' + epoch))
    plt.show()
    f = plt.gcf()  # 获取当前图像
    # f.savefig()
    f.clear()  # 释放内存


def visualize_LossGraph(config, log_dir, i, x,
                        t_loss, v_loss, te_loss,
                        train_loss, valid_loss, test_loss,
                        train_loss_lines, val_loss_lines, te_loss_lines):
    x.append(i + 1)  # 此步为更新迭代步数
    t_loss.append(train_loss)
    v_loss.append(valid_loss)
    te_loss.append(test_loss)
    try:
        train_loss_lines.remove(train_loss_lines[0])  # 移除上一步曲线
        val_loss_lines.remove(val_loss_lines[0])
        te_loss_lines.remove(te_loss_lines[0])
    except Exception:
        pass
    train_loss_lines = plt.plot(x, t_loss, 'r', lw=5)  # lw为曲线宽度
    val_loss_lines = plt.plot(x, v_loss, 'b', lw=5)
    te_loss_lines = plt.plot(x, te_loss, 'g', lw=5)
    plt.title("loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend(["train_loss",
                "val_loss", "test_loss"])
    plt.pause(0.15)  # 图片停留0.1s
    if i == config['epochs']:
        f = plt.gcf()  # 获取当前图像
        f.savefig(log_dir + '/loss_fig_epoch{}.png'.format(i))
        f.clear()  # 释放内存
    return x, t_loss, v_loss, te_loss, train_loss_lines, val_loss_lines, te_loss_lines


def visualize(about_epoch, embbeding_y, right_label, val_idx, config, draw_data):
    if about_epoch in draw_data['visualization_epoch']:
        about_epoch = about_epoch.split('_')[2]
        embbeding_y = embbeding_y[val_idx]
        draw_data['withorwithoutHG'] = 'without latter part'
        match config['latterPart']:
            case 'HGNNP':
                draw_data['withorwithoutHG'] = 'with HyperGraph+'
            case 'HGNN':
                draw_data['withorwithoutHG'] = 'with HyperGraph'
            case 'nothing':
                draw_data['withorwithoutHG'] = 'without HyperGraph'
            case _:
                draw_data['withorwithoutHG'] = ''

        if draw_data['draw_class'] == 'all':
            visualize_PCA(about_epoch, embbeding_y, right_label, draw_data, config['class_num'])
            visualize_t_SNE(about_epoch, embbeding_y, right_label, draw_data, config['class_num'])
            draw_umap_plot(about_epoch, embbeding_y, right_label, draw_data, config['class_num'])
        else:
            visualize_PCA(about_epoch, embbeding_y, right_label, draw_data, 2)
            visualize_t_SNE(about_epoch, embbeding_y, right_label, draw_data, 2)
            draw_umap_plot(about_epoch, embbeding_y, right_label, draw_data, 2)
        if draw_data['visualization_coefficient']:
            # 可视化 计算相关指数
            embbeding_y = embbeding_y.cpu().detach().numpy()
            # 将onehot转化为数字
            right_label = np.argmax(right_label, axis=1)
            print(f"可视化_轮廓系数 : {silhouette_score(embbeding_y, right_label)}")
            print(f"可视化_方差比准则 : {calinski_harabasz_score(embbeding_y, right_label)}")
            print(f"可视化_DB指数 : {davies_bouldin_score(embbeding_y, right_label)}")
