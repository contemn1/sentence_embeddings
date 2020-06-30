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
KEY_LIST = ["GLOVE", "$c^0$", "$c^{0:1}$", "$c^{0:2}$", "$c^{0:3}$", "$c^{0:4}$", "$c^{0:5}$", "$c^{0:6}$",
            "SkipThought", "QuickThought", "InferSentV1", "InferSentV2", "GenSen",
            "UniversalSentenceDAN", "UniversalSentenceTransformer",
            "BERT-BASE-AVG", "BERT-BASE-CLS", "BERT-LARGE-AVG", "BERT-LARGE-CLS",
            "XLNET-BASE-AVG", "XLNET-BASE-CLS", "XLNET-LARGE-AVG", "XLNET-LARGE-CLS",
            "ROBERTA-BASE-AVG", "ROBERTA-BASE-CLS", "ROBERTA-LARGE-AVG", "ROBERTA-LARGE-CLS",
            "SBERT-BASE-AVG", "SBERT-BASE-CLS", "SBERT-LARGE-AVG", "SBERT-LARGE-CLS",
            "SRoBERTa-BASE-AVG", "SRoBERTa-LARGE-AVG"]
ALIAS_DICT = {"$c^0$": "DCT(k=0)", "$c^{0:1}$": "DCT(k=1)", "$c^{0:2}$": "DCT(k=2)", "$c^{0:3}$": "DCT(k=3)",
              "$c^{0:4}$": "DCT(k=4)", "$c^{0:5}$": "DCT(k=5)", "$c^{0:6}$": "DCT(k=6)",
              "UniversalSentenceDAN": "USE-DAN", "UniversalSentenceTransformer": "USE-Transformer"}

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
        if key == "UniversalSentenceDAN":
            key = "USE\\_D"

        if key == "UniversalSentenceTransformer":
            key = "USE\\_T"

        return key, Statistics(sub_dict)
    else:
        all_data = [(sub_key, Statistics(value)) for sub_key, value in sub_dict.items()]
        all_data.sort(key=lambda x: x[1].precision_add_masked, reverse=True)
        key = u"DCT" if key == "CDT" else all_data[0][0]
        if key == "DCT":
            print(all_data[0][0])
        return key, all_data[0][1]


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
    with open(overall_dict_path, "w+") as overall_out:
        json.dump(overall_dict, overall_out)
        '''
        json.dump(semantic_overall, semantic_out)
        json.dump(syntactic_overall, syntactic_out)
        '''


def cos_add_metric(intra_similarity, inter_similarity, intra_diagonal):
    return inter_similarity - intra_similarity + intra_diagonal


def cos_mul_metric(intra_similarity, inter_similarity, intra_diagonal, eps=0.001):
    return np.divide(inter_similarity * intra_diagonal, intra_similarity + eps)


def calculate_score_matrix(embedding_x, embedding_y, embedding_z, calculation_metric=cos_add_metric):
    intra_similarity = cosine_similarity(embedding_x, embedding_z)
    inter_similarity = cosine_similarity(embedding_y, embedding_z)
    intra_diagonal = np.expand_dims(intra_similarity.diagonal(), axis=0)
    return calculation_metric(intra_similarity, inter_similarity, intra_diagonal)


def get_result_matrix(hypo_embedding, premise_embedding, neg_embedding_list, calculation_metric=cos_add_metric):
    positive_score_matrix = calculate_score_matrix(hypo_embedding, premise_embedding, premise_embedding, calculation_metric)
    positive_score_matrix = np.expand_dims(positive_score_matrix, axis=2)
    negative_score_matrix_list = [
        np.expand_dims(calculate_score_matrix(hypo_embedding, premise_embedding,  neg, calculation_metric), axis=2, ) for neg in
        neg_embedding_list]
    all_score_matrix_list = [positive_score_matrix] + negative_score_matrix_list
    concatenated_score_matrix = np.concatenate(all_score_matrix_list, axis=2)
    selected_element = np.argmax(concatenated_score_matrix, axis=2)
    return selected_element


def get_selected_element(pretrained_embeddings, key, metric):
    embedding = pretrained_embeddings.get(key)[()]
    indices = np.arange(start=0, stop=embedding.shape[0])
    hypo_embedding = embedding[indices[indices % 7 == 0]]
    premise_embedding = embedding[indices[indices % 7 == 1]]
    neg_embedding_list = [embedding[indices[indices % 7 == idx]] for idx in range(2, 7)]
    selected_element = get_result_matrix(hypo_embedding, premise_embedding, neg_embedding_list, metric)
    return selected_element


def output_experiment_result(category_name: str, result_dict: dict):
    new_result_dict = {}
    value_length = 0
    for key in KEY_LIST:
        accuracy_list = result_dict[key]
        value_length = len(accuracy_list)
        res_key = key if not key in ALIAS_DICT else ALIAS_DICT[key]
        if not res_key[:3] == "DCT" and not res_key[-3:] in {"AVG", "CLS"}:
            new_result_dict[res_key] = accuracy_list
        else:
            target_key = "DCT" if res_key[:3] == "DCT" else res_key[:-4]
            current_list = new_result_dict.pop(target_key, list())
            accuracy_list.append(res_key)
            current_list = accuracy_list if not current_list or current_list[0] < accuracy_list[0] else current_list
            new_result_dict[target_key] = current_list
    new_result_dict = {value.pop() if len(value) == value_length + 1 else key: value for key, value in new_result_dict.items()}

    print("\n")
    print("\\begin{table}")
    print("\\caption {{Experiment Results on {0} Sentence Analogy Pairs}}".format(category_name.capitalize()))
    print("\\centering")
    print("\\resizebox{0.75\columnwidth}{!}{")
    print("\\begin{tabular}{|l|l|l|}")
    print("\\hline")
    print("\t & 3CosADD  \t & 3CosMul  \\\\ \\hline")
    for res_key, accuracy_list in new_result_dict.items():
        print("{0} \t & {1:.4f} \t & {2:.4f}   \\\\ \\hline".format(res_key, *accuracy_list))
    print("\\end{tabular}")
    print("}")
    print("\\label{table:}")
    print("\\end{table}")


def analyze_relation_based_analogy(file_path: str):
    pretrained_embeddings = h5py.File(file_path, "r")
    metric_dict = {"3CosADD": cos_add_metric, "3CosMul": cos_mul_metric}
    result_dict = {}
    for name, ele in metric_dict.items():
        res_dict = {}
        for key in pretrained_embeddings.keys():
            if key == "Sentences":
                continue
            selected_element = get_selected_element(pretrained_embeddings, key, ele)
            accuracy_list = list()
            all_questions = 0
            for index in range(6):
                result_matrix = selected_element == index
                np.fill_diagonal(result_matrix, False)
                all_questions = result_matrix.shape[0] * (result_matrix.shape[0] - 1)
                accuracy_list.append(np.sum(result_matrix))

            accuracy_list.append(all_questions)
            res_dict[key] = accuracy_list

        result_dict[name] = res_dict
    return result_dict


def calculate_relation_based_analogy(file_path: str):
    pretrained_embeddings = h5py.File(file_path, "r")
    metric_list = [cos_add_metric, cos_mul_metric]
    result_dict = {}
    for key in pretrained_embeddings.keys():
        if key == "Sentences":
            continue
        accuracy_list = []
        all_questions  = 0
        for ele in metric_list:
            selected_element = get_selected_element(pretrained_embeddings, key, ele)
            result_matrix = selected_element == 0
            np.fill_diagonal(result_matrix, False)
            all_questions = result_matrix.shape[0] * (result_matrix.shape[0] - 1)
            accuracy_list.append(np.sum(result_matrix))

        accuracy_list.append(all_questions)
        result_dict[key] = accuracy_list
    return result_dict


def word_analogy_percategory_test():
    root_dir = "/home/zxj/Data/relation_based_analogy/output"
    category_name = "passivization"
    file_path = "new_{0}_analogy_embeddings.h5".format(category_name)
    pretrained_embeddings = h5py.File(os.path.join(root_dir, file_path), "r")
    embedding_dict = dict()
    for key in pretrained_embeddings.keys():
        if key == "Sentences":
            continue
        embedding = pretrained_embeddings[key]
        indices = [ele for ele in range(embedding.shape[0]) if ele % 7 == 0 or ele % 7 == 1]
        embedding_dict[key] = embedding[indices]

    print("\n")
    print("\\begin{table}")
    print("\\caption {{Experiment Results on {0} Sentence Analogy Pairs}}".format(category_name.capitalize()))
    print("\\centering")
    print("\\resizebox{0.75\columnwidth}{!}{")
    print("\\begin{tabular}{|l|l|l|l|l|}")
    print("\\hline")
    print("\t & 3CosADD  \t& 3CosMul \\\\ \\hline")

    for key in embedding_dict.keys():
        embeddings = embedding_dict[key]
        similarity_matrix = cosine_similarity(embeddings)
        similar_sentences_add, prediction_result_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=False)
        true_positive_add = np.sum(prediction_result_add).item()
        all_questions = prediction_result_add.shape[0] * prediction_result_add.shape[1]
        precision_add = true_positive_add / all_questions
        masked_similar_sentences_add, masked_prediction_result_add = calculate_3cos_add_result(
            similarity_matrix, use_mask=True)
        masked_true_positive_add = np.sum(masked_prediction_result_add).item()
        masked_precision_add = masked_true_positive_add / all_questions
        similar_sentences_mul, prediction_result_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=False)
        true_positive_mul = np.sum(prediction_result_mul).item()
        precision_mul = true_positive_mul / all_questions
        masked_similar_sentences_mul, masked_prediction_result_mul = calculate_3cos_mul_result(
            similarity_matrix, use_mask=True)
        masked_true_positive_mul = np.sum(masked_prediction_result_mul).item()
        masked_precision_mul = masked_true_positive_mul / all_questions
        print("{0} \t & {1:.4f} \t & {2:.4f} \t & {3:.4f} \t & {4:.4f} \t \\\\ \\hline".format(key, precision_add, masked_precision_add, precision_mul, masked_precision_mul))


    print("\\end{tabular}")
    print("}")
    print("\\label{table:}")
    print("\\end{table}")


def merge_dict(dict_list):
    all_result_dict = defaultdict(list)
    for ele in dict_list:
        for key, value in ele.items():
            all_result_dict[key].append(value)

    for key, value in all_result_dict.items():
        value_array = np.array(value).sum(axis=0)
        all_result_dict[key] = (value_array[:-1] / value_array[-1]).tolist()

    return all_result_dict


if __name__ == '__main__':
    category_name_list = ["entailment", "contradiction", "passivization", "sub_clause", "adjective"]
    for category_name in category_name_list:
        file_path = "/home/zxj/Data/relation_based_analogy/output/{0}_analogy_embeddings.h5".format(category_name)
        result_dict = calculate_relation_based_analogy(file_path)
        result_dict = {key: [ele / value[-1] for ele in value[:-1]] for key, value in result_dict.items()}
        output_experiment_result(category_name, result_dict)