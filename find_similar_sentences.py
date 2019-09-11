import io
import json
import os
import re
import string
import pickle
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from encode_sentences import infersent_encoder
from io_util import output_iterator


def output_similarity_matrix():
    dict_path = "/home/zxj/Data/multinli_1.0/word-pairs-per-category.json"
    sentence_templates = "I'm not sure if they can travel to {0}"
    pairs_dict = json.load(open(dict_path))
    country_dict = pairs_dict[": capital-common-countries"]
    sentence_list = []
    for key, value in country_dict.items():
        first = sentence_templates.format(string.capwords(key))
        second = sentence_templates.format(string.capwords(value))
        sentence_list.append(first)
        sentence_list.append(second)


    model_path = "/media/zxj/sent_embedding_data/infersent/infersent2.pkl"
    word2vec_path = "/media/zxj/sent_embedding_data/infersent/crawl-300d-2M.vec"

    sent_encoder = infersent_encoder(model_path, word2vec_path, use_cuda=True)

    sent_embeddings = sent_encoder(sentence_list)

    similarity_matix = cosine_similarity(sent_embeddings)

    np.save("similarity_matrix", similarity_matix)


def calculate_3cos_add_result(matrix, use_mask=False):
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
        most_similar_sentence_list.append(most_similar_sentence)
        prediction_result_list.append(prediction_result)

    most_similar_sentence_all = np.vstack(most_similar_sentence_list)
    prediction_result_all = np.vstack(prediction_result_list)
    true_positive = np.sum(prediction_result_all)
    precision = float(true_positive) / (prediction_result_all.shape[0] * prediction_result_all.shape[1])

    return most_similar_sentence_all, prediction_result_all, precision


def calculate_3cos_mul_result(matrix, use_mask=False):
    odd_cols = matrix[:, ::2]
    even_cols = matrix[:, 1::2]
    diff = np.divide(even_cols, (odd_cols + 0.001)).T
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
        result[:, index] = -10
        most_similar_sentence = np.argmax(result, axis=1)
        prediction_result = (most_similar_sentence == index + 1)
        most_similar_sentence_list.append(most_similar_sentence)
        prediction_result_list.append(prediction_result)

    most_similar_sentence_all = np.vstack(most_similar_sentence_list)
    prediction_result_all = np.vstack(prediction_result_list)
    true_positive = np.sum(prediction_result_all)
    precision = float(true_positive) / (prediction_result_all.shape[0] * prediction_result_all.shape[1])

    return most_similar_sentence_all, prediction_result_all, precision


def main():
    file_name_list = ["currency_words_embeddings.h5"]
    root_dir = "/media/zxj/sent_embedding_data/output"
    key_list = [u'GenSen', u'InferSentV2', u'QuickThought', u'SkipThought', u'UniversalSentence']
    for name in file_name_list:
        file_path = os.path.join(root_dir, name)
        print (name)
        with h5py.File(file_path, "r") as file:
            for key in key_list:
                embeddings = file.get(key)[()]
                similarity_matrix = cosine_similarity(embeddings)
                similar_sentences_add, prediction_result_add, precision_add = calculate_3cos_add_result(
                    similarity_matrix)
                similar_sentences_mul, prediction_result_mul, precision_mul = calculate_3cos_mul_result(
                    similarity_matrix)
                print (similar_sentences_add)
                print ("{0} && {1:.3f} && {2:.3f}".format(key, precision_add, precision_mul))


if __name__ == '__main__':
    pretrained_emb = "/home/zxj/Data/"
    pretrained_embeddings = h5py.File(pretrained_emb)
    print (pretrained_embeddings['InferSentV1'][()])
    print (pretrained_embeddings['InferSentV2'][()])