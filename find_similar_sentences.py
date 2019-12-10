from __future__ import print_function

import json
import os
import re
from collections import defaultdict
from itertools import product
from typing import Dict, List, Set

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name_list = ["BERT", "XLNET", "ROBERTA", "SBERT", "SRoBERTa"]
suffix_name_list = ["BASE", "LARGE"]
model_set = set(["-".join(ele) for ele in product(model_name_list, suffix_name_list)])
model_set.add("CDT")
KEY_LIST = ["GLOVE", "DCT", "InferSentV1", "InferSentV2",
            "GenSen", "SkipThought", "QuickThought",
            "UniversalSentenceDAN", "UniversalSentenceTransformer",
            "BERT-BASE", "BERT-LARGE", "XLNET-BASE", "XLNET-LARGE",
            "ROBERTA-BASE", "ROBERTA-LARGE", "SBERT-BASE", "SBERT-LARGE",
            "SRoBERTa-BASE", "SRoBERTa-LARGE"]

name_dict = {"all": "all_precision_result.json",
             "semantic": "semantic_precision_result.json",
             "syntactic": "syntactic_precision_result.json",
             }


def calculate_3cos_add_result(matrix: np.ndarray, use_mask=False):
    odd_cols = matrix[:, ::2]
    even_cols = matrix[:, 1::2]
    diff = (even_cols - odd_cols).T
    if use_mask:
        for i in range(0, diff.shape[0], 2):
            diff[i][2 * i] = - 100
            diff[i][2 * i + 1] = -100

    most_similar_sentence_list = []
    prediction_result_list = []
    for index in range(0, matrix.shape[1], 2):
        current_row = np.expand_dims(matrix[index], axis=0)
        new_diff = np.delete(diff, index // 2, axis=0)
        result = current_row + new_diff
        if use_mask:
            result[:, index] = -100
        most_similar_sentence = np.argmax(result, axis=1)
        prediction_result = (most_similar_sentence == index + 1)
        most_similar_sentence_list.append(most_similar_sentence.tolist())
        prediction_result_list.append(prediction_result)

    prediction_result_all = np.vstack(prediction_result_list)
    return most_similar_sentence_list, prediction_result_all


def calculate_3cos_mul_result(matrix, use_mask=False):
    odd_cols = matrix[:, ::2]
    even_cols = matrix[:, 1::2]
    diff = np.divide(even_cols, (odd_cols + 0.00001)).T
    if use_mask:
        for i in range(0, diff.shape[0], 2):
            diff[i][2 * i] = - 100
            diff[i][2 * i + 1] = -100

    most_similar_sentence_list = []
    prediction_result_list = []
    for index in range(0, matrix.shape[1], 2):
        current_row = np.expand_dims(matrix[index], axis=0)
        new_diff = np.delete(diff, index // 2, axis=0)
        result = current_row * new_diff
        if use_mask:
            result[:, index] = -10
        most_similar_sentence = np.argmax(result, axis=1)
        prediction_result = (most_similar_sentence == index + 1)
        most_similar_sentence_list.append(most_similar_sentence.tolist())
        prediction_result_list.append(prediction_result)

    prediction_result_all = np.vstack(prediction_result_list)
    return most_similar_sentence_list, prediction_result_all


def output_precision(input_dict: Dict[str, int], category_name, title_template: str, key_list: List[str] = None):
    print("\n")
    print("\\begin{table}")
    print(title_template.format(category=category_name))
    print("\\centering")
    print("\\begin{tabular}{|l|l|l|l|l|}")
    print("\\hline")
    print("\t & 3CosADD Fair \t& 3CosADD  \t& 3CosMul Fair \t &3CosMul  \\\\ \\hline")
    if not key_list:
        key_list = input_dict.keys()
    for key in key_list:
        precision_add, masked_precision_add, precision_mul, masked_precision_mul = input_dict[key]
        print("{0} \t & {1:.4f} \t & {2:.4f}  \t & {3:.4f}  \t & {4:.4f} \t \\\\ \\hline".format(key, precision_add,
                                                                                                 masked_precision_add,
                                                                                                 precision_mul,
                                                                                                 masked_precision_mul))
    print("\\end{tabular}")
    print("\\end{table}")


def calculate_prediction_result(category_name: str, h5_dataset, key_list: List[str] = None):
    precision_dict = defaultdict(dict)
    result_dict = {}
    if not key_list:
        key_list = [ele for ele in h5_dataset.keys() if ele != "Sentences"]

    for key in key_list:
        embeddings = h5_dataset.get(key)[()]
        similarity_matrix = cosine_similarity(embeddings)
        similar_sentences_add, prediction_result_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=False)
        true_positive_add = np.sum(prediction_result_add).item()
        all_questions = prediction_result_add.shape[0] * prediction_result_add.shape[1]
        masked_similar_sentences_add, masked_prediction_result_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=True)

        masked_true_positive_add = np.sum(masked_prediction_result_add).item()
        similar_sentences_mul, prediction_result_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=False)
        true_positive_mul = np.sum(prediction_result_mul).item()
        masked_similar_sentences_mul, masked_prediction_result_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=True)
        masked_true_positive_mul = np.sum(masked_prediction_result_mul).item()

        result_dict[key] = {"masked": {"similar_sentences_add": masked_similar_sentences_add,
                                       "similar_sentences_mul": masked_similar_sentences_mul},
                            "traditional": {"similar_sentences_add": similar_sentences_add,
                                            "similar_sentences_mul": similar_sentences_mul}}

        precision_result = {"true_positive_add_masked": masked_true_positive_add,
                            "true_positive_mul_masked": masked_true_positive_mul,
                            "true_positive_add_tra": true_positive_add,
                            "true_positive_mul_tra": true_positive_mul,
                            "all_questions": all_questions,
                            }
        if key[-3:] == "AVG" or key[-3:] == "CLS" and key[:5] != "GLOVE":
            new_key = key[: -4]
            precision_dict[new_key][key] = precision_result

        elif re.search(r"\$c", key):
            precision_dict["DCT"][key] = precision_result

        else:
            precision_dict[key] = precision_result

    return result_dict, precision_dict


def main():
    root_dir = "/home/zxj/Data/sent_embedding_data/output"
    file_name_list = os.listdir(root_dir)
    title_template = "\\caption {{ Experiment Results on {category} sentence analogy pairs }}"
    category_name_list = [re.sub(r"_pairs_embeddings.h5", "", name) for name in file_name_list]
    all_prediction_results = {}
    all_precision_results = {}
    for file_name, category in zip(file_name_list, category_name_list):
        pretrained_embeddings = h5py.File(os.path.join(root_dir, file_name), "r")
        category = re.sub("_", "\\_", category)
        prediction_result_dict, precision_dict = calculate_prediction_result(category, pretrained_embeddings)
        all_prediction_results[category] = prediction_result_dict
        all_precision_results[category] = precision_dict


def flatten_dict(model_dict):
    traditonal_dict = model_dict.pop("traditional")
    model_dict["true_positive_mul_tra"] = traditonal_dict["true_positive_mul"]
    model_dict["true_positive_add_tra"] = traditonal_dict["true_positive_add"]
    model_dict["all_questions"] = traditonal_dict["all_questions"]
    masked_dict = model_dict.pop("masked")
    model_dict["true_positive_mul_masked"] = masked_dict["true_positive_mul"]
    model_dict["true_positive_add_masked"] = masked_dict["true_positive_add"]


def merge_dict(*dicts):
    new_dict = defaultdict(int)
    for sub_dict in dicts:
        for key, value in sub_dict.items():
            new_dict[key] += value
    return new_dict


def generate_overall_results(input_dict):
    all_model_dict = dict()
    for category, category_dict in input_dict.items():
        for name, model_dict in category_dict.items():
            if name not in model_set:
                all_model_dict[name] = merge_dict(all_model_dict[name],
                                                  model_dict) if name in all_model_dict else model_dict
            elif name not in all_model_dict:
                all_model_dict[name] = model_dict
            else:
                for key, value in all_model_dict[name].items():
                    all_model_dict[name][key] = merge_dict(model_dict[key], value)
    return all_model_dict


class Statistics(object):
    def __init__(self, object_dict):
        self.all_questions = object_dict["all_questions"]
        self.true_positive_add_masked = object_dict["true_positive_add_masked"]
        self.true_positive_mul_masked = object_dict["true_positive_mul_masked"]
        self.true_positive_add_tra = object_dict["true_positive_add_tra"]
        self.true_positive_mul_tra = object_dict["true_positive_mul_tra"]

    @property
    def precision_add_masked(self):
        return float(self.true_positive_add_masked) / self.all_questions

    @property
    def precision_add_tra(self):
        return float(self.true_positive_add_tra) / self.all_questions

    @property
    def precision_mul_maksed(self):
        return float(self.true_positive_mul_masked) / self.all_questions

    @property
    def precision_mul_tra(self):
        return float(self.true_positive_mul_tra) / self.all_questions

    @property
    def precision_tuple(self):
        return self.precision_mul_tra, self.precision_add_masked, self.precision_mul_tra, self.precision_mul_maksed


def transform_dict_items(key: str, sub_dict: dict, model_name_set: Set[str]):
    if key not in model_name_set:
        return key, Statistics(sub_dict)
    else:
        all_data = [(sub_key, Statistics(value)) for sub_key, value in sub_dict.items()]
        all_data.sort(key=lambda x: x[1].precision_add_masked, reverse=True)
        key = u"DCT" if key == "CDT" else key
        return key, all_data[0][1]


def generate_latex_from_dict(json_dict, title_template, category_name, key_list=None):
    result_list = [transform_dict_items(key, sub_dict, model_set) for key, sub_dict in json_dict.items()]
    result_dict = {key: value.precision_tuple for key, value in result_list}
    output_precision(result_dict, category_name, title_template, key_list)


def overall_results_main():
    root_dir = "/home/zxj/Data/sent_embedding_data"
    file_path = os.path.join(root_dir, "precision_result_flatten.json")
    json_dict = json.load(open(file_path, mode="r"))
    '''
    semantic_cateogry = {"capital\_world", "city\_in\_state", "family", "currency", "capital\_country"}
    semantic_dict = {key: value for key, value in json_dict.items() if key in semantic_cateogry}
    syntactic_dict = {key: value for key, value in json_dict.items() if key not in semantic_cateogry}
    semantic_overall = generate_overall_results(semantic_dict)
    syntactic_overall = generate_overall_results(syntactic_dict)
    semantic_path = os.path.join(root_dir, "semantic_precision_result.json")
    syntactic_path = os.path.join(root_dir, "syntactic_precision_result.json")

    '''
    overall_dict = generate_overall_results(json_dict)
    overall_dict_path = os.path.join(root_dir, "all_precision_result.json")
    with  open(overall_dict_path, "w+") as overall_out:
        json.dump(overall_dict, overall_out)
        '''
        json.dump(semantic_overall, semantic_out)
        json.dump(syntactic_overall, syntactic_out)
        '''


if __name__ == '__main__':
    root_dir = "/home/zxj/Data/relation_based_analogy/output"
