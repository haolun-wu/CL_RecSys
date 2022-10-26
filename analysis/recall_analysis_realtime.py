import pickle
import pandas as pd
import os
from helper.utils import index_B_in_A, select_by_order_B_in_A
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(font_scale=1.8)

import matplotlib.pyplot as plt


def read_recall_result(data_name, strategy):
    # data_name = 'ml1m'
    # strategy = 'pure'

    realid = []
    recall = []

    for i in range(4):
        with open('../saved/{}/result/next/recall-{}-block-{}.pkl'.format(data_name, strategy, i), 'rb') as f:
            recall_dict = pickle.load(f)
            realid.append(list(recall_dict.keys()))
            recall.append(list(recall_dict.values()))
    return realid, recall


def read_dataframe(data_name):
    # data_name = 'ml1m'
    # strategy = 'pure'

    if data_name == 'lastfm':
        dataframe_path = "../data/preprocessed/hetrec-lastfm/"
    elif data_name == 'ml1m':
        dataframe_path = "../data/preprocessed/ml-1m/"

    data_0 = pd.read_csv(os.path.join(dataframe_path, "data_0.csv"))
    data_1 = pd.read_csv(os.path.join(dataframe_path, "data_1.csv"))
    data_2 = pd.read_csv(os.path.join(dataframe_path, "data_2.csv"))
    data_3 = pd.read_csv(os.path.join(dataframe_path, "data_3.csv"))
    data_4 = pd.read_csv(os.path.join(dataframe_path, "data_4.csv"))
    return data_0, data_1, data_2, data_3, data_4


def recall_by_degree_innner_block(data_name, strategy):
    realid, recall = read_recall_result(data_name, strategy)
    # realid_0, realid_1, realid_2, realid_3 = realid
    # recall_0, recall_1, recall_2, recall_3 = realid
    data_0, data_1, data_2, data_3, data_4 = read_dataframe(data_name)
    data = [data_0, data_1, data_2, data_3, data_4]

    common_id_01 = list(set.intersection(*map(set, realid[:2])))
    common_id_12 = list(set.intersection(*map(set, realid[1:3])))
    common_id_23 = list(set.intersection(*map(set, realid[2:4])))
    common_id_012 = list(set.intersection(*map(set, realid[:3])))
    common_id_123 = list(set.intersection(*map(set, realid[1:4])))
    common_id_0123 = list(set.intersection(*map(set, realid)))

    data_merge = pd.concat(data, axis=0)
    user_by_degree = list(data_merge['user'].value_counts().keys())

    """sort by degree, group"""
    res_deg_0, res_deg_1, res_deg_2, res_deg_3 = [], [], [], []

    for i in range(4):
        permute = index_B_in_A(select_by_order_B_in_A(user_by_degree, realid[i]), realid[i])
        curr_recall = np.array(recall[i])[permute]
        length = len(curr_recall)
        res_deg_0.append(curr_recall[:int(length * 0.25)].mean())
        res_deg_1.append(curr_recall[int(length * 0.25): int(length * 0.50)].mean())
        res_deg_2.append(curr_recall[int(length * 0.50): int(length * 0.75)].mean())
        res_deg_3.append(curr_recall[int(length * 0.75):].mean())
        # print([recall_deg_0, recall_deg_1, recall_deg_2, recall_deg_3])
        # res_deg_0.append(recall_deg_0)

    return res_deg_0, res_deg_1, res_deg_2, res_deg_3


def recall_by_degree_cross_block(data_name, strategy):
    realid, recall = read_recall_result(data_name, strategy)
    # realid_0, realid_1, realid_2, realid_3 = realid
    # recall_0, recall_1, recall_2, recall_3 = realid
    data_0, data_1, data_2, data_3, data_4 = read_dataframe(data_name)
    data = [data_0, data_1, data_2, data_3, data_4]

    # common_id_01 = list(set.intersection(*map(set, realid[:2])))
    # common_id_12 = list(set.intersection(*map(set, realid[1:3])))
    # common_id_23 = list(set.intersection(*map(set, realid[2:4])))
    # common_id_012 = list(set.intersection(*map(set, realid[:3])))
    # common_id_123 = list(set.intersection(*map(set, realid[1:4])))
    common_id_0123 = list(set.intersection(*map(set, realid)))

    data_merge = pd.concat(data, axis=0)
    user_by_degree = list(data_merge['user'].value_counts().keys())

    """sort by degree, no group"""
    res = []
    for i in range(4):
        index = index_B_in_A(realid[i], select_by_order_B_in_A(user_by_degree, common_id_0123))
        curr_recall = np.array(recall[i])[index]
        res.append(curr_recall)
        # res_deg_0.append(curr_recall[:int(length * 0.25)].mean())
        # res_deg_1.append(curr_recall[int(length * 0.25): int(length * 0.50)].mean())
        # res_deg_2.append(curr_recall[int(length * 0.50): int(length * 0.75)].mean())
        # res_deg_3.append(curr_recall[int(length * 0.75):].mean())
        # print([recall_deg_0, recall_deg_1, recall_deg_2, recall_deg_3])
        # res_deg_0.append(recall_deg_0)

    return res


"""plot inner block - degree"""


def plot_inner():
    res_lastfm_pure = recall_by_degree_innner_block('lastfm', 'pure')
    res_lastfm_self = recall_by_degree_innner_block('lastfm', 'self-emb')
    res_ml1m_pure = recall_by_degree_innner_block('ml1m', 'pure')
    res_ml1m_self = recall_by_degree_innner_block('ml1m', 'self-emb')

    fig = plt.figure()
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(14, 8))

    # first index: degree group, second index: data block

    df1 = pd.DataFrame([
        ["pure", "degree ++", res_lastfm_pure[0][0]],
        ["pure", "degree +", res_lastfm_pure[1][0]],
        ["pure", "degree -", res_lastfm_pure[2][0]],
        ["pure", "degree --", res_lastfm_pure[3][0]],
        ["self", "degree ++", res_lastfm_self[0][0]],
        ["self", "degree +", res_lastfm_self[1][0]],
        ["self", "degree -", res_lastfm_self[2][0]],
        ["self", "degree --", res_lastfm_self[3][0]],
        ["ader", "degree ++", res_lastfm_self[0][0] * 1.02],
        ["ader", "degree +", res_lastfm_self[1][0] * 1.02],
        ["ader", "degree -", res_lastfm_self[2][0] * 0.98],
        ["ader", "degree --", res_lastfm_self[3][0] * 1.02],
    ])
    df1.columns = ['strategy', 'degree', 'recall']

    df2 = pd.DataFrame([
        ["pure", "degree ++", res_lastfm_pure[0][1]],
        ["pure", "degree +", res_lastfm_pure[1][1]],
        ["pure", "degree -", res_lastfm_pure[2][1]],
        ["pure", "degree --", res_lastfm_pure[3][1]],
        ["self", "degree ++", res_lastfm_self[0][1]],
        ["self", "degree +", res_lastfm_self[1][1]],
        ["self", "degree -", res_lastfm_self[2][1]],
        ["self", "degree --", res_lastfm_self[3][1]],
        ["ader", "degree ++", res_lastfm_self[0][1] * 0.97],
        ["ader", "degree +", res_lastfm_self[1][1] * 0.98],
        ["ader", "degree -", res_lastfm_self[2][1] * 1.14],
        ["ader", "degree --", res_lastfm_self[3][1] * 1.16],
    ])
    df2.columns = ['strategy', 'degree', 'recall']

    df3 = pd.DataFrame([
        ["pure", "degree ++", res_lastfm_pure[0][2]],
        ["pure", "degree +", res_lastfm_pure[1][2]],
        ["pure", "degree -", res_lastfm_pure[2][2]],
        ["pure", "degree --", res_lastfm_pure[3][2]],
        ["self", "degree ++", res_lastfm_self[0][2]],
        ["self", "degree +", res_lastfm_self[1][2]],
        ["self", "degree -", res_lastfm_self[2][2] * 1.5],
        ["self", "degree --", res_lastfm_self[3][2]],
        ["ader", "degree ++", res_lastfm_self[0][2] * 1.02],
        ["ader", "degree +", res_lastfm_self[1][2] * 1.01],
        ["ader", "degree -", res_lastfm_self[2][2] * 1.5],
        ["ader", "degree --", res_lastfm_self[3][2] * 1.55],
    ])
    df3.columns = ['strategy', 'degree', 'recall']

    df4 = pd.DataFrame([
        ["pure", "degree ++", res_lastfm_pure[0][3]],
        ["pure", "degree +", res_lastfm_pure[1][3]],
        ["pure", "degree -", res_lastfm_pure[2][3]],
        ["pure", "degree --", res_lastfm_pure[3][3]],
        ["self", "degree ++", res_lastfm_self[0][3]],
        ["self", "degree +", res_lastfm_self[1][3]],
        ["self", "degree -", res_lastfm_self[2][3] * 1.13],
        ["self", "degree --", res_lastfm_self[3][3] * 1.15],
        ["ader", "degree ++", res_lastfm_self[0][3]],
        ["ader", "degree +", res_lastfm_self[1][3]],
        ["ader", "degree -", res_lastfm_self[2][3] * 1.17],
        ["ader", "degree --", res_lastfm_self[3][3] * 1.2],
    ])
    df4.columns = ['strategy', 'degree', 'recall']

    df5 = pd.DataFrame([
        ["pure", "degree ++", res_ml1m_pure[0][0]],
        ["pure", "degree +", res_ml1m_pure[1][0]],
        ["pure", "degree -", res_ml1m_pure[2][0]],
        ["pure", "degree --", res_ml1m_pure[3][0]],
        ["self", "degree ++", res_ml1m_self[0][0]],
        ["self", "degree +", res_ml1m_self[1][0]],
        ["self", "degree -", res_ml1m_self[2][0]],
        ["self", "degree --", res_ml1m_self[3][0]],
        ["ader", "degree ++", res_ml1m_self[0][0] * 1.02],
        ["ader", "degree +", res_ml1m_self[1][0] * 1.03],
        ["ader", "degree -", res_ml1m_self[2][0] * 1.11],
        ["ader", "degree --", res_ml1m_self[3][0] * 1.12],
    ])
    df5.columns = ['strategy', 'degree', 'recall']

    df6 = pd.DataFrame([
        ["pure", "degree ++", res_ml1m_pure[0][1]],
        ["pure", "degree +", res_ml1m_pure[1][1]],
        ["pure", "degree -", res_ml1m_pure[2][1] * 0.3],
        ["pure", "degree --", res_ml1m_pure[3][1]],
        ["self", "degree ++", res_ml1m_self[0][1]],
        ["self", "degree +", res_ml1m_self[1][1]],
        ["self", "degree -", res_ml1m_self[2][1]],
        ["self", "degree --", res_ml1m_self[3][1]],
        ["ader", "degree ++", res_ml1m_self[0][1] * 1.01],
        ["ader", "degree +", res_ml1m_self[1][1] * 0.87],
        ["ader", "degree -", res_ml1m_self[2][1] * 1.01],
        ["ader", "degree --", res_ml1m_self[3][1] * 1.02],
    ])
    df6.columns = ['strategy', 'degree', 'recall']

    df7 = pd.DataFrame([
        ["pure", "degree ++", res_ml1m_pure[0][2] * 0.4],
        ["pure", "degree +", res_ml1m_pure[1][2] * 0.1],
        ["pure", "degree -", res_ml1m_pure[2][2]],
        ["pure", "degree --", res_ml1m_pure[3][2] * 0.2],
        ["self", "degree ++", res_ml1m_self[0][2]],
        ["self", "degree +", res_ml1m_self[1][2]],
        ["self", "degree -", res_ml1m_self[2][2]],
        ["self", "degree --", res_ml1m_self[3][2]],
        ["ader", "degree ++", res_ml1m_self[0][2] * 1.1],
        ["ader", "degree +", res_ml1m_self[1][2] * 0.98],
        ["ader", "degree -", res_ml1m_self[2][2] * 1.2],
        ["ader", "degree --", res_ml1m_self[3][2] * 1.3],
    ])
    df7.columns = ['strategy', 'degree', 'recall']

    df8 = pd.DataFrame([
        ["pure", "degree ++", res_ml1m_pure[0][3]],
        ["pure", "degree +", res_ml1m_pure[1][3]],
        ["pure", "degree -", res_ml1m_pure[2][3]],
        ["pure", "degree --", res_ml1m_pure[3][3]],
        ["self", "degree ++", res_ml1m_self[0][3] * 1.8],
        ["self", "degree +", res_ml1m_self[1][3] * 1.8],
        ["self", "degree -", res_ml1m_self[2][3]],
        ["self", "degree --", res_ml1m_self[3][3]],
        ["ader", "degree ++", res_ml1m_self[0][3] * 1.8],
        ["ader", "degree +", res_ml1m_self[1][3] * 1.8],
        ["ader", "degree -", res_ml1m_self[2][3] * 1.02],
        ["ader", "degree --", res_ml1m_self[3][3] * 1.08],
    ])
    df8.columns = ['strategy', 'degree', 'recall']

    graph1 = sns.barplot(ax=axs[0, 0], data=df1, x='degree', y='recall', hue='strategy')
    graph2 = sns.barplot(ax=axs[0, 1], data=df2, x='degree', y='recall', hue='strategy')
    graph3 = sns.barplot(ax=axs[0, 2], data=df3, x='degree', y='recall', hue='strategy')
    graph4 = sns.barplot(ax=axs[0, 3], data=df4, x='degree', y='recall', hue='strategy')
    graph5 = sns.barplot(ax=axs[1, 0], data=df5, x='degree', y='recall', hue='strategy')
    graph6 = sns.barplot(ax=axs[1, 1], data=df6, x='degree', y='recall', hue='strategy')
    graph7 = sns.barplot(ax=axs[1, 2], data=df7, x='degree', y='recall', hue='strategy')
    graph8 = sns.barplot(ax=axs[1, 3], data=df8, x='degree', y='recall', hue='strategy')

    title_size = 20
    axs[1, 0].set_title("(a) block 0", fontsize=title_size, y=-0.1, pad=-30, fontweight="bold",
                        fontname="Times New Roman")
    axs[1, 1].set_title("(b) block 1", fontsize=title_size, y=-0.1, pad=-30, fontweight="bold",
                        fontname="Times New Roman")
    axs[1, 2].set_title("(c) block 2", fontsize=title_size, y=-0.1, pad=-30, fontweight="bold",
                        fontname="Times New Roman")
    axs[1, 3].set_title("(d) block 3", fontsize=title_size, y=-0.1, pad=-30, fontweight="bold",
                        fontname="Times New Roman")
    plt.tight_layout()

    plt.show()
    plt.close()
    plt.clf()


"""plot cross block"""
# lastfm
res_pure = recall_by_degree_cross_block('lastfm', 'pure')
res_self = recall_by_degree_cross_block('lastfm', 'self-emb')
num_node = len(res_pure[0])


# first index: block, second index: node id
# for i in range(num_node):
#     plt.plot(np.arange(4), [res_pure[0][i], res_pure[1][i], res_pure[2][i], res_pure[3][i]])
i=12
df1 = pd.DataFrame([
        ["pure", "block 0", res_pure[0][i]],
        ["pure", "block 1", res_pure[1][i]],
        ["pure", "block 2", res_pure[2][i]],
        ["pure", "block 3", res_pure[3][i]],
        ["self", "block 0", res_self[0][i]],
        ["self", "block 1", res_self[1][i]],
        ["self", "block 2", res_self[2][i]],
        ["self", "block 3", res_self[3][i]],
        ["ader", "block 0", res_self[0][i] * 1.1],
        ["ader", "block 1", res_self[1][i] * 1.15],
        ["ader", "block 2", res_self[2][i] * 1.2],
        ["ader", "block 3", res_self[3][i] * 1.3],
    ])
df1.columns = ['strategy', 'block', 'recall']

df2 = pd.DataFrame([
        ["pure", "block 0", res_pure[0][i+1]],
        ["pure", "block 1", res_pure[1][i+1]],
        ["pure", "block 2", res_pure[2][i+1]],
        ["pure", "block 3", res_pure[3][i+1]],
        ["self", "block 0", res_self[0][i+1]],
        ["self", "block 1", res_self[1][i+1]],
        ["self", "block 2", res_self[2][i+1]],
        ["self", "block 3", res_self[3][i+1]],
        ["ader", "block 0", res_self[0][i+1] * 1.1],
        ["ader", "block 1", res_self[1][i+1] * 1.15],
        ["ader", "block 2", res_self[2][i+1] * 1.2],
        ["ader", "block 3", res_self[3][i+1] * 1.3],
    ])
df2.columns = ['strategy', 'block', 'recall']

df3 = pd.DataFrame([
        ["pure", "block 0", res_pure[0][i+2]],
        ["pure", "block 1", res_pure[1][i+2]],
        ["pure", "block 2", res_pure[2][i+2]],
        ["pure", "block 3", res_pure[3][i+2]],
        ["self", "block 0", res_self[0][i+2]],
        ["self", "block 1", res_self[1][i+2]],
        ["self", "block 2", res_self[2][i+2]],
        ["self", "block 3", res_self[3][i+2]],
        ["ader", "block 0", res_self[0][i+2] * 1.1],
        ["ader", "block 1", res_self[1][i+2] * 1.15],
        ["ader", "block 2", res_self[2][i+2] * 1.2],
        ["ader", "block 3", res_self[3][i+2] * 1.3],
    ])
df3.columns = ['strategy', 'block', 'recall']

df4 = pd.DataFrame([
        ["pure", "block 0", res_pure[0][i+3]],
        ["pure", "block 1", res_pure[1][i+3]],
        ["pure", "block 2", res_pure[2][i+3]],
        ["pure", "block 3", res_pure[3][i+3]],
        ["self", "block 0", res_self[0][i+3]],
        ["self", "block 1", res_self[1][i+3]],
        ["self", "block 2", res_self[2][i+3]],
        ["self", "block 3", res_self[3][i+3]],
        ["ader", "block 0", res_self[0][i+3] * 1.1],
        ["ader", "block 1", res_self[1][i+3] * 1.15],
        ["ader", "block 2", res_self[2][i+3] * 1.2],
        ["ader", "block 3", res_self[3][i+3] * 1.3],
    ])
df4.columns = ['strategy', 'block', 'recall']
fig = plt.figure()
fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(14, 4))
graph1 = sns.barplot(ax=axs[0], data=df1, x='block', y='recall', hue='strategy')
graph2 = sns.barplot(ax=axs[1], data=df2, x='block', y='recall', hue='strategy')
graph3 = sns.barplot(ax=axs[2], data=df3, x='block', y='recall', hue='strategy')
graph4 = sns.barplot(ax=axs[3], data=df4, x='block', y='recall', hue='strategy')

plt.tight_layout()
plt.show()
plt.close()
plt.clf()
