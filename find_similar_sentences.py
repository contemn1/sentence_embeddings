from __future__ import print_function

import os
import re

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_3cos_add_result(matrix, use_mask=False):
    odd_cols = matrix[:, ::2]
    even_cols = matrix[:, 1::2]
    diff = (even_cols - odd_cols).T

    if use_mask:
        for i in range(0, diff.shape[0], 2):
            diff[i][2 * i] = - 100
            diff[i][2 * i + 1] = -100

    result_list = []
    for index in range(0, matrix.shape[1], 2):
        current_row = np.expand_dims(matrix[index], axis=0)
        new_diff = np.delete(diff, index // 2, axis=0)
        result = current_row + new_diff
        if use_mask:
            result[:, index] = -100

        result_list.append(np.expand_dims(result, 0))

    return np.vstack(result_list)


def calculate_3cos_mul_result(matrix, use_mask=False):
    odd_cols = matrix[:, ::2]
    even_cols = matrix[:, 1::2]
    diff = np.divide(even_cols, (odd_cols + 0.00001)).T
    print(diff.shape)
    if use_mask:
        for i in range(0, diff.shape[0], 2):
            diff[i][2 * i] = - 100
            diff[i][2 * i + 1] = -100

    result_list = []
    for index in range(0, matrix.shape[1], 2):
        current_row = np.expand_dims(matrix[index], axis=0)
        new_diff = np.delete(diff, index // 2, axis=0)
        result = current_row * new_diff
        if use_mask:
            result[:, index] = -10

        result_list.append(np.expand_dims(result, 0))

    return np.vstack(result_list)


def output_precision(key_list, precision_dict, category_name, title_template):
    print("\n")
    print("\\begin{table}")
    print(title_template.format(category=category_name))
    print("\\centering")
    print("\\begin{tabular}{|l|l|l|l|l|}")
    print("\\hline")
    print("\t & 3CosADD Fair \t& 3CosADD  \t& 3CosMul Fair \t &3CosMul  \\\\ \\hline")
    for key in key_list:
        precision_add, masked_precision_add, precision_mul, masked_precision_mul = precision_dict[key_list]
        print("{0} \t & {1:.4f} \t & {2:.4f}  \t & {3:.4f}  \t & {4:.4f} \t \\\\ \\hline".format(key, precision_add,
                                                                                                 masked_precision_add,
                                                                                                 precision_mul,
                                                                                                 masked_precision_mul))
    print("\\end{tabular}")
    print("\\end{table}")


def main(category_name, h5_dataset, key_list=None):
    if not key_list:
        key_list = [ele for ele in h5_dataset.keys() if ele != "Sentences"]

    for key in key_list:
        embeddings = h5_dataset.get(key)[()]
        similarity_matrix = cosine_similarity(embeddings)
        similar_sentences_add, prediction_result_add, precision_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=False)
        masked_similar_sentences_add, masked_prediction_result_add, masked_precision_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=True)

        masked_similar_sentences_mul, masked_prediction_result_mul, masked_precision_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=True)
        similar_sentences_mul, prediction_result_mul, precision_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=False)
        total_num = similar_sentences_add.shape[0] * similar_sentences_add.shape[1]


if __name__ == '__main__':
    root_dir = "/home/zxj/Data/sent_embedding_data/output"
    file_name_list = os.listdir(root_dir)
    title_template = "\\caption {{ Experiment Results on {category} sentence analogy pairs }}"
    category_name_list = [re.sub(r"_pairs_embeddings.h5", "", name) for name in file_name_list]
    for file_name, category in zip(file_name_list, category_name_list):
        if category == "currency":
            pretrained_embeddings = h5py.File(os.path.join(root_dir, file_name), "r")
            category = re.sub("_", "\\_", category)
            main(category, pretrained_embeddings)
            print("\n")
