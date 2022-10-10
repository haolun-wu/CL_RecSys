# import pandas as pd
#
# def generate_inverse_mapping(data_list):
#     ds_matrix_mapping = dict()
#     for inner_id, true_id in enumerate(data_list):
#         print("inner_id:{}, true_id:{}".format(inner_id, true_id))
#         ds_matrix_mapping[true_id] = inner_id
#     return ds_matrix_mapping
#
# def convert_unique_idx(df, column_name):
#     column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
#     df[column_name] = df[column_name].apply(column_dict.get)
#     df[column_name] = df[column_name].astype('int')
#     assert df[column_name].min() == 0
#     assert df[column_name].max() == len(column_dict) - 1
#     return df, column_dict
#
# data_user = [768, 28771, 12821, 21, 28771, 768]
# data_item = [213, 325, 213, 47, 567, 325]
# data_frame = {'user':data_user, 'item':data_item}
# df = pd.DataFrame(data=data_frame)
# print("df:", df)
#
# df, user_dict = convert_unique_idx(df, "user")
# df, item_dict = convert_unique_idx(df, "item")
# print("df:", df)
# print("user_dict:", user_dict)
# print("item_dict:", item_dict)

a = {10:0, 23:1, 47:2}
b = {29:0, 47:1, 38:2}

def dict_merge_extend(dict1, dict2):
    merged_dict = dict1.copy()
    count = [*dict1.values()][-1]
    for key, values in dict2.items():
        if key not in dict1:
            merged_dict[key] = count+1
            count += 1
    return merged_dict

def separete_intersect_dicts(dict1, dict2):
    return_dict1, return_dict2 = {}, {}
    for key, values in dict1.items():
        if key in dict2:
            return_dict1[key] = dict1[key]
            return_dict2[key] = dict2[key]
    return return_dict1, return_dict2

# c = dict_merge_extend(a, b)
#
# print(a)
# print(b)
# print(c)
return_dict1, return_dict2 = separete_intersect_dicts(a, b)
print(return_dict1)
print(return_dict2)