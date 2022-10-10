import sys
import os

import pandas

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle as pkl
import pandas as pd
import torch
from copy import deepcopy

[sys.path.append(i) for i in ['.', '..']]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import pickle


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


def data_statistics(data_list):
    for data in data_list:
        # print("------------")
        print("{}:".format(data[0]))
        print("# interact:{}, # user:{}, # item:{}".format(data[1].shape[0], len(data[1]['user'].unique()),
                                                           len(data[1]['item'].unique())))
        # min_counts = 5
        # counts = data[1]['user'].value_counts()
        # print("# user (interaction > 5):{}".format(counts[counts >= min_counts].index.shape[0]))


class lastfm(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = data_dir

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='\t',
                         engine='python',
                         names=['user', 'item', 'tag', 'time'],
                         usecols=['user', 'item', 'time'],
                         skiprows=1).drop_duplicates()

        return df


class Data(object):
    def __init__(self, data_dir, val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5, seed=0):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.user_filter = user_filter
        self.item_filter = item_filter
        origin_data_path = os.path.join(data_dir, "hetrec-lastfm/user_taggedartists-timestamps.dat")
        pro_data_path = os.path.join(data_dir, "preprocessed/hetrec-lastfm/overall.csv")
        pro_data_block_path = os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_0.csv")

        if not os.path.exists(pro_data_path):
            df = lastfm(origin_data_path).load()

            df['rate'] = 1

            print("start generate unique idx")
            df['user'] = df['user'].astype('category').cat.codes
            df['item'] = df['item'].astype('category').cat.codes
            df['time'] = df['time'].astype('category').cat.codes

            df = df.sort_values(by=['time'])
            df = df.reset_index().drop(['index'], axis=1)

            df.to_csv(pro_data_path)
            self.num_user = len(df['user'].unique())
            self.num_item = len(df['item'].unique())
            self.num_interac = df.shape[0]

            print('all_user:{}, all_item:{}, all_interac:{}'.format(self.num_user, self.num_item, self.num_interac))
        else:
            df = pandas.read_csv(pro_data_path)
            print('all_user:{}, all_item:{}, all_interac:{}'.format(1892, 12523, 87061))

        if not os.path.exists(pro_data_block_path):
            num_interact = 87061
            data_0 = df.loc[0:int(num_interact * 0.6)].reset_index().drop(['index'], axis=1)
            data_1 = df.loc[int(num_interact * 0.6):int(num_interact * 0.7)].reset_index().drop(['index'], axis=1)
            data_2 = df.loc[int(num_interact * 0.7):int(num_interact * 0.8)].reset_index().drop(['index'], axis=1)
            data_3 = df.loc[int(num_interact * 0.8):int(num_interact * 0.9)].reset_index().drop(['index'], axis=1)
            data_4 = df.loc[int(num_interact * 0.9):].reset_index().drop(['index'], axis=1)
            data_0.to_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_0.csv"))
            data_1.to_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_1.csv"))
            data_2.to_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_2.csv"))
            data_3.to_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_3.csv"))
            data_4.to_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_4.csv"))
        else:
            data_0 = pandas.read_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_0.csv"))
            data_1 = pandas.read_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_1.csv"))
            data_2 = pandas.read_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_2.csv"))
            data_3 = pandas.read_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_3.csv"))
            data_4 = pandas.read_csv(os.path.join(data_dir, "preprocessed/hetrec-lastfm/data_4.csv"))

        data_full = [data_0, data_1, data_2, data_3, data_4]
        # self.user_dict, self.item_dict, self.train_matrix, self.test_matrix, self.val_matrix, \
        # self.train_set, self.test_set, self.val_set = [], [], [], [], [], [], [], []
        # self.preprocess_data_block(data_0)
        # self.preprocess_data_block(data_1)
        # self.preprocess_data_block(data_2)
        # self.preprocess_data_block(data_3)
        # self.preprocess_data_block(data_4)

        self.data_full_dict = {}
        for i in range(len(data_full)):
            print("------------")
            print("Block:{}".format(i))
            self.data_full_dict[i] = self.preprocess_data_block(data_full[i], time_block=i)

        user0 = set(list(self.data_full_dict[0][0].keys()))
        user1 = set(list(self.data_full_dict[1][0].keys()))
        user2 = set(list(self.data_full_dict[2][0].keys()))
        user3 = set(list(self.data_full_dict[3][0].keys()))
        user4 = set(list(self.data_full_dict[4][0].keys()))

        # print("original:", len(list(set(user0) & set(user1))))
        #
        # print("common:", len(list(set(user0) & set(user1))))
        # print("common:", len(list(set(user1) & set(user2))))
        # print("common:", len(list(set(user2) & set(user3))))
        # print("common:", len(list(set(user3) & set(user4))))
        # self.plot_statistics(user0, user1, user2, user3, user4)

        # print("U-I:")
        # print(np.sum([len(x) for x in self.train_set_UI]),
        #       np.sum([len(x) for x in self.val_set_UI]),
        #       np.sum([len(x) for x in self.test_set_UI]))
        # density = float(
        #     np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
        #         [len(x) for x in self.test_set_UI])) / self.num_user / self.num_item
        # print("density:{:.2%}".format(density))
        #
        # avg_deg = np.sum([len(x) for x in self.train_set_UI]) / self.num_user
        # avg_deg *= 1 / (1 - test_ratio)
        # print('Avg. degree U-I graph: ', avg_deg)

    def preprocess_data_block(self, data_block, time_block):
        data_block = data_block[["user", "item", "rate"]].drop_duplicates()
        if time_block == 0:
            user_filter, item_filter = 5, 5
        else:
            user_filter, item_filter = 3, 5
        data_block = self.remove_infrequent_users(data_block, user_filter)
        data_block = self.remove_infrequent_items(data_block, item_filter)
        data_block = self.remove_infrequent_users(data_block, user_filter)
        data_block, user_dict = self.convert_unique_idx(data_block, "user")
        data_block, item_dict = self.convert_unique_idx(data_block, "item")
        num_user = data_block['user'].max() + 1
        num_item = data_block['item'].max() + 1
        print("# user:{}, # item:{}, # interact:{}".format(num_user, num_item, data_block.shape[0]))
        UI_matrix = scipy.sparse.csr_matrix(
            (np.array(data_block['rate']), (np.array(data_block['user']), np.array(data_block['item']))))

        if time_block == 0:
            if self.val_ratio:
                self.val_ratio = self.val_ratio / (1 - self.test_ratio)
        else:
            # if self.val_ratio:
            #     self.val_ratio = self.val_ratio / (1 - self.test_ratio)
            self.val_ratio = 0
            self.test_ratio = 0.5

        train_matrix, test_matrix, val_matrix, train_set, test_set, val_set \
            = self.create_train_test_split(UI_matrix, test_ratio=self.test_ratio, val_ratio=self.val_ratio, seed=0)

        print("# train:{}, # val:{}, # test:{}".format(np.sum([len(x) for x in train_set]),
                                                       np.sum([len(x) for x in val_set]),
                                                       np.sum([len(x) for x in test_set])))
        density = float(
            np.sum([len(x) for x in train_set]) + np.sum([len(x) for x in val_set]) + np.sum(
                [len(x) for x in test_set])) / num_user / num_item
        print("density:{:.2%}".format(density))
        #
        # avg_deg = np.sum([len(x) for x in self.train_set_UI]) / self.num_user
        # avg_deg *= 1 / (1 - test_ratio)
        # print('Avg. degree U-I graph: ', avg_deg)
        # self.user_dict.append(user_dict)
        # self.item_dict.append(item_dict)
        # self.train_matrix.append(train_matrix)
        # self.test_matrix.append(test_matrix)
        # self.val_matrix.append(val_matrix)
        # self.train_set.append(train_set)
        # self.test_set.append(test_set)
        # self.val_set.append(val_set)

        return user_dict, item_dict, train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

    def plot_statistics(self, set0, set1, set2, set3, set4):
        import pandas as pd
        import seaborn as sns
        import numpy as np
        sns.set(font_scale=1.8)

        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (10, 10)
        import matplotlib.gridspec as gridspec
        #
        gs = gridspec.GridSpec(1, 8)
        ax1 = plt.subplot(gs[0, :8])

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
        legend_size = 15
        font_size = 20
        title_size = 28

        df1 = pd.DataFrame([
            ["new to all", "block0", len(set0)],
            ["new to last", "block0", len(set0)],
            ["number", "block0", len(set0)],
            ["new to all", "block1", len(set1 - set0)],
            ["new to last", "block1", len(set1 - set0)],
            ["number", "block1", len(set1)],
            ["new to all", "block2", len(set2 - (set0 | set1))],
            ["new to last", "block2", len(set2 - set1)],
            ["number", "block2", len(set2)],
            ["new to all", "block3", len(set3 - (set0 | set1 | set2))],
            ["new to last", "block3", len(set3 - set2)],
            ["number", "block3", len(set3)],
            ["new to all", "block4", len(set4 - (set0 | set1 & set2 | set3))],
            ["new to last", "block4", len(set4 - set3)],
            ["number", "block4", len(set4)],
        ])
        x_label = ["block0", "block1", "block2", "block3", "block4"]

        df1.columns = ['Model', '', 'Relative Performance']

        graph1 = sns.barplot(ax=ax1, data=df1, x='', y='Relative Performance', hue='Model')
        graph1.axhline(1, color='r', linestyle='--')

        ax1.legend(loc=(0.03, 1.01), fontsize=legend_size, ncol=3)
        # ax1.set_ylim(0.8, 1.1)

        ax1.set_xticks(np.arange(6))
        ax1.set_xticklabels(x_label, rotation=60, fontsize=font_size)

        # ax1.set_title("(a) HetRec-Del", fontsize=title_size, y=-0.1, pad=-90, fontweight="bold",
        #               fontname="Times New Roman")

        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.5, hspace=0.7)
        # plt.savefig('./fig_pdf/ISA.pdf', bbox_inches='tight')

        plt.show()
        plt.close()
        plt.clf()

    def generate_inverse_mapping(self, data_list):
        ds_matrix_mapping = dict()
        for inner_id, true_id in enumerate(data_list):
            ds_matrix_mapping[true_id] = inner_id
        return ds_matrix_mapping

    def split_data_randomly(self, user_records, val_ratio, test_ratio, seed=0):
        train_set = []
        test_set = []
        val_set = []
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

            if val_ratio:
                tmp_train_sample, tmp_val_sample = train_test_split(tmp_train_sample, test_size=val_ratio,
                                                                    random_state=seed)

            if val_ratio:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_val_sample:
                        train_sample.append(place)

                val_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_train_sample:
                        val_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample and place not in tmp_val_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                val_set.append(val_sample)
                test_set.append(test_sample)

            else:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample:
                        train_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                test_set.append(test_sample)

        return train_set, test_set, val_set

    def create_train_test_split(self, rating_df, val_ratio=None, test_ratio=0.1, seed=0):
        data_set = []
        for item_ids in rating_df:
            data_set.append(item_ids.indices.tolist())
        # data_set = self.data_set

        train_set, test_set, val_set = self.split_data_randomly(data_set, val_ratio=val_ratio, test_ratio=test_ratio,
                                                                seed=seed)
        train_matrix = self.generate_rating_matrix(train_set, rating_df.shape[0], rating_df.shape[1])
        test_matrix = self.generate_rating_matrix(test_set, rating_df.shape[0], rating_df.shape[1])
        val_matrix = self.generate_rating_matrix(val_set, rating_df.shape[0], rating_df.shape[1])

        return train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

    def generate_rating_matrix(self, train_matrix, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        # triplet = []
        for user_id, article_list in enumerate(train_matrix):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)
                # triplet.append((int(user_id), int(article), float(1)))

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        return rating_matrix

    def convert_to_inner_index(self, user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping = self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def convert_to_lists(self, user_records):
        inner_user_records = []
        user_records = user_records.A

        for user_id in range(user_records.shape[0]):
            item_list = np.where(user_records[user_id] != 0)[0].tolist()
            inner_user_records.append(item_list)

        return inner_user_records

    def remove_infrequent_items(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['item'].value_counts()
        df = df[df['item'].isin(counts[counts >= min_counts].index)]

        # print("items with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_users(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['user'].value_counts()
        df = df[df['user'].isin(counts[counts >= min_counts].index)]

        # print("users with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_tags(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['tag'].value_counts()
        df = df[df['tag'].isin(counts[counts >= min_counts].index)]

        print("tags with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def convert_unique_idx(self, df, column_name):
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
        df[column_name] = df[column_name].apply(column_dict.get)
        df[column_name] = df[column_name].astype('int')
        assert df[column_name].min() == 0
        assert df[column_name].max() == len(column_dict) - 1
        return df, column_dict


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    # data_dir = "/Users/haolunwu/Research_project/CL_RecSys/data/"
    data_dir = "C:/Users/eq22858/Documents/GitHub/CL_RecSys/data/"
    data_generator = Data(data_dir, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter)
