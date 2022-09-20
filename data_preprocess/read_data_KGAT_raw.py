import sys
import os
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
import warnings

warnings.filterwarnings('ignore')


def generate_IT_matrix(df):
    df_IT = df[['item', 'tag']]
    df_IT = df_IT.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    # print("df_genre:", df_genre)
    # df_IT = pd.concat([pd.Series(row['item'], row['genre'].split(',')) for _, row in df_genre.iterrows()]).reset_index()
    df_IT = pd.DataFrame([[i, t] for i, T in df_IT.values for t in T], columns=df_IT.columns)
    # c = df_IT.columns
    # df_IT[[c[0], c[1]]] = df_IT[[c[1], c[0]]]
    df_IT.columns = ['item', 'tag']

    return df_IT


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class Read_kgat(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = data_dir

    def load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()
        inter_list = []

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            inter_list.append(pos_ids)

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict, inter_list

    def load(self):
        # load user_list, item_list, entity_list
        data_user = pd.read_csv(self.fpath + '/user_list.txt', sep=" ", header=None, error_bad_lines=False)
        data_item = pd.read_csv(self.fpath + '/item_list.txt', sep=" ", header=None, error_bad_lines=False)
        data_entity = pd.read_csv(self.fpath + '/entity_list.txt', sep=" ", header=None, error_bad_lines=False)
        num_user = len(data_user.index) - 1
        num_item = len(data_item.index) - 1
        num_tag = len(data_entity.index) - num_item - 1
        # print("num of user:", num_user)
        # print("num of item:", num_item)
        # print("num of tag:", num_tag)

        # Load data ui_interaction
        inter_mat_train, u_dict_train, inter_list_train = self.load_ratings(self.fpath + '/train.txt')

        # print("inter_mat_test:", inter_mat_test, inter_mat_test.shape)
        # print("u_dict_test:", u_dict_test[0])
        # print("inter_list_test:", inter_list_test[0], len(inter_list_test))

        # for key, value in u_dict_train.items():
        #     if key in u_dict_test:
        #         value.extend(u_dict_test[key])  # Surprised there is empty in test set
        u_items = u_dict_train.items()

        df_UI = pd.DataFrame(list(u_items))
        df_UI.columns = ['user', 'item']
        df_UI = df_UI.explode('item').reset_index(drop=True)

        # Load data it_interaction
        df_IT = pd.read_csv(self.fpath + '/kg_final.txt', sep=" ", header=None)
        df_IT.columns = ["item", "relation", "tag"]
        df_IT = pd.concat(
            [df_IT[["item", "tag"]], df_IT[["item", "tag"]].rename(columns={'item': 'tag', 'tag': 'item'})])
        df_IT = df_IT.loc[(df_IT["item"] < num_item) & (df_IT["tag"] >= num_item)]
        df_IT["tag"] = df_IT["tag"] - num_item
        df_IT = df_IT.sort_values(by=['tag']).reset_index(drop=True)
        df_IT = df_IT.reset_index().drop(['index'], axis=1)
        return df_UI, df_IT, num_user, num_item, num_tag


class Data(object):
    def __init__(self, data_dir, data_name, val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5, seed=0):

        file_path = os.path.join("/home/haolun/projects/UIT/data_preprocess/preprocessed/",
                                 '{}_raw.pkl'.format(data_name))
        array_path = os.path.join("/home/haolun/projects/UIT/data_preprocess/preprocessed/",
                                  '{}_richitem.npy'.format(data_name))
        #
        if os.path.exists(file_path) and os.path.exists(array_path):
            df = pd.read_pickle(file_path)
            rich_items = np.load(array_path)
            # print("load saved:\n", df)
        else:
            df_UI, df_IT, num_user, num_item, num_tag = Read_kgat(data_dir).load()
            df_IT = df_IT.groupby(['item'])['tag'].apply(list)
            rich_items = df_IT[df_IT.to_frame()['tag'].apply(lambda x: len(set(x)) >= 5)].to_frame().index.tolist()
            rich_items = np.array(rich_items)
            self.rich_items = rich_items
            print("rich_items:", rich_items, len(rich_items))

            print("merging...")
            df = pd.merge(df_UI, df_IT, on='item')

            df['rate'] = 1
            # df.drop_duplicates(subset=['user', 'item'], keep='last', inplace=True)
            # df = df.reset_index().drop(['index'], axis=1)

            df.to_pickle(file_path)
            np.save(array_path, rich_items)

        # print("df:", df)
        IT_df = df[['item', 'tag']].explode('tag').reset_index(drop=True)
        # print("IT_df:", IT_df)
        IT_df['tag_new'] = IT_df['tag'].astype('category').cat.codes
        IT_df['rate'] = 1
        IT_df.drop_duplicates(subset=['item', 'tag_new'], keep='last', inplace=True)
        IT_df = IT_df.reset_index().drop(['index'], axis=1)

        # df_tag_mapping = pd.merge(df_tag_name, IT_df, on='tag')
        # df_tag_mapping = df_tag_mapping[['tag_new', 'name']].drop_duplicates().reset_index().drop(['index'], axis=1)
        # print("df_tag_mapping:", df_tag_mappig)
        # df_tag_mapping.to_pickle("/home/haolunn/projects/UIT/saved/lastfm/tagname_mapping.pkl")

        self.num_user = len(df['user'].unique())
        self.num_item = len(df['item'].unique())
        self.num_tag = len(IT_df['tag_new'].unique())

        print('user:{}, item:{}, tag:{}'.format(self.num_user, self.num_item, self.num_tag))

        if data_name == 'last-fm':
            tmp_df = df['user'].value_counts().to_frame().apply(lambda x: x >= 3)  # .index.tolist()
            rich_users = tmp_df.loc[tmp_df['user'] == True].index.tolist()
            self.rich_users = np.array(rich_users)
            print("users:(item>3 in train):", len(rich_users))

            # # new user mapping
            # all_users = np.arange(self.num_user)
            # permute_users = np.concatenate((rich_users, np.setdiff1d(all_users, rich_users)))
            # new_user_mapping = dict(zip(permute_users, all_users))

            # new item mapping
            all_items = np.arange(self.num_item)
            permute_items = np.concatenate((rich_items, np.setdiff1d(all_items, rich_items)))
            new_item_mapping = dict(zip(permute_items, all_items))

            # df['user'] = df['user'].map(new_user_mapping)
            df['item'] = df['item'].map(new_item_mapping)
            IT_df['item'] = IT_df['item'].map(new_item_mapping)

        self.UI_matrix = scipy.sparse.csr_matrix((np.array(df['rate']), (np.array(df['user']), np.array(df['item']))))

        self.IT_matrix = scipy.sparse.csr_matrix(
            (np.array(IT_df['rate']), (np.array(IT_df['item']), np.array(IT_df['tag_new']))))

        if data_name == 'last-fm':
            self.UI_matrix = self.UI_matrix[:, np.arange(len(rich_items))]
            self.IT_matrix = self.IT_matrix[np.arange(len(rich_items)), :]

        _, _, self.test_set_UI = Read_kgat(data_dir).load_ratings(data_dir + '/test.txt')

        # here the test_set is the given test_set in the original dataset
        if data_name != 'last-fm':
            self.train_matrix_UI, self.val_matrix_UI, _, \
            self.train_set_UI, self.val_set_UI, _ = self.create_train_test_split(self.UI_matrix,
                                                                                 test_ratio=val_ratio,
                                                                                 val_ratio=0, seed=0)
        else:
            # self.train_set_UI
            _, _, self.train_set_UI = Read_kgat(data_dir).load_ratings(data_dir + '/train.txt')
            # self.train_matrix_UI = self.generate_rating_matrix(self.train_set_UI, self.UI_matrix.shape[0],
            #                                                    self.UI_matrix.shape[1])
            self.train_matrix_UI = self.generate_rating_matrix(self.train_set_UI, self.num_user, self.num_item)[:,
                                   np.arange(len(rich_items))]
            self.val_set_UI = self.test_set_UI

        # if data_name == 'last-fm':
        #     self.test_set_UI = [self.test_set_UI[i] for i in rich_users][:len(rich_users)]

        if val_ratio:
            val_ratio = val_ratio / (1 - test_ratio)

        self.train_matrix_IT, self.test_matrix_IT, self.val_matrix_IT, \
        self.train_set_IT, self.test_set_IT, self.val_set_IT = self.create_train_test_split(self.IT_matrix,
                                                                                            test_ratio=test_ratio,
                                                                                            val_ratio=val_ratio, seed=0)

        tag_count_train = np.asarray(self.train_matrix_IT.sum(1)).flatten()
        # item_list = np.array(item_tag_count.index.to_list())
        # tag_count = np.array(item_tag_count.item.to_list())
        # index = np.where(tag_count >= 10)[0][-1]
        self.poor_items = np.where(tag_count_train < 10)[0]
        self.poor_items_tags = tag_count_train[self.poor_items]
        print("items:(tag<10 in train)", self.poor_items, len(self.poor_items))

        self.rich_items = rich_items
        print("items:(tag>5 in train-val-test)", len(self.rich_items))
        if data_name == 'last-fm':
            self.num_item = len(rich_items)

        print("U-I: train-val-test:", np.sum([len(x) for x in self.train_set_UI]),
              np.sum([len(x) for x in self.val_set_UI]),
              np.sum([len(x) for x in self.test_set_UI]))
        density = float(
            np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
                [len(x) for x in self.test_set_UI])) / self.num_user / self.num_item
        print("density:{:.2%}".format(density))

        print("I-T: train-val-test:", np.sum([len(x) for x in self.train_set_IT]),
              np.sum([len(x) for x in self.val_set_IT]),
              np.sum([len(x) for x in self.test_set_IT]))
        density = float(
            np.sum([len(x) for x in self.train_set_IT]) + np.sum([len(x) for x in self.val_set_IT]) + np.sum(
                [len(x) for x in self.test_set_IT])) / self.num_tag / self.num_item
        print("density:{:.2%}".format(density))

        avg_deg = (np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
            [len(x) for x in self.test_set_UI])) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

        avg_deg = (np.sum([len(x) for x in self.train_set_IT]) + np.sum([len(x) for x in self.val_set_IT]) + np.sum(
            [len(x) for x in self.test_set_IT])) / self.num_item
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree I-T graph: ', avg_deg)

        """
        Distribution & statistics
        """
        # from plt.plot_distribution import draw_distribution
        # draw_distribution(df[['user', 'item']], 'user', 'item', "lastfm")
        # draw_distribution(df[['user', 'item']], 'item', 'user', "lastfm")
        # draw_distribution(IT_df, 'item', 'tag', "lastfm")
        # draw_distribution(IT_df, 'tag', 'item', "lastfm")

        tmp_count = df['user'].value_counts().to_frame().apply(lambda x: x >= 0)  # .index.tolist()
        user_sorted = np.array(tmp_count.loc[tmp_count['user'] == True].index.tolist())
        users_c1 = user_sorted[:int(self.num_user * 0.2)]
        users_c2 = user_sorted[int(self.num_user * 0.2):int(self.num_user * 0.8)]
        users_c3 = user_sorted[int(self.num_user * 0.8):]
        print("users_c1:{}, users_c2:{}, users_c3:{}".format(len(users_c1), len(users_c2), len(users_c3)))

        tmp_count = IT_df['item'].value_counts().to_frame().apply(lambda x: x >= 0)  # .index.tolist()
        item_sorted = np.array(tmp_count.loc[tmp_count['item'] == True].index.tolist())
        items_c1 = item_sorted[:int(self.num_item * 0.2)]
        items_c2 = item_sorted[int(self.num_item * 0.2):int(self.num_item * 0.8)]
        items_c3 = item_sorted[int(self.num_item * 0.8):]
        print("items_c1:{}, items_c2:{}, items_c3:{}".format(len(items_c1), len(items_c2), len(items_c3)))

        self.user_index = [users_c1, users_c2, users_c3]
        self.item_index = [items_c1, items_c2, items_c3]

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

    def create_train_test_split(self, rating_df, val_ratio=None, test_ratio=0.2, seed=0):
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

        print("items with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_users(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['user'].value_counts()
        df = df[df['user'].isin(counts[counts >= min_counts].index)]

        print("users with < {} interactoins are removed".format(min_counts))
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
    parser.add_argument('--data_name', type=str, default="last-fm")
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    data_dir = '../data/KGAT/{}'.format(parser.data_name)
    data_generator = Data(data_dir, parser.data_name, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter)

    # print("data_genrator:", data_generator)
