import pickle
import pandas as pd
import os

data_name = 'ml1m'
strategy = 'pure'


def read_recall_result():
    real_id = []
    recall = []

    for i in range(4):
        with open('../saved/{}/result/recall-{}-block-{}.pkl'.format(data_name, strategy, i), 'rb') as f:
            recall_dict = pickle.load(f)
            real_id.append(list(recall_dict.keys()))
            recall.append(list(recall_dict.values()))
    return real_id, recall


def read_dataframe():
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


real_id, recall = read_recall_result()
data_0, data_1, data_2, data_3, data_4 = read_dataframe()
