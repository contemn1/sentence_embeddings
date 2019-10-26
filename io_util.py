from __future__ import print_function, unicode_literals

import io
import logging
import sys

import h5py


def read_file(file_path, encoding="utf-8", preprocess=lambda x: x):
    try:
        with io.open(file_path, encoding=encoding) as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def output_iterator(file_path, output_list, process=lambda x: x):
    try:
        with io.open(file_path, mode="w+", encoding="utf-8") as file:
            for line in output_list:
                file.write(process(line) + "\n")
    except IOError as error:
        logging.error("Failed to open file {0}".format(error))
        sys.exit(1)


def get_embedding_dict(embedding_path, word_dict):
    # create word_vec with glove vectors
    word_vec = {}
    with h5py.File(open(embedding_path), 'r') as f:
        for idx, word in enumerate(f["words_flatten"][()].split("\n")):
            if word in word_dict:
                word_vec[word] = f["embedding"][idx]

    logging.info('Found {0}(/{1}) words with glove vectors'.format(
        len(word_vec), len(word_dict)))

    return word_vec


def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    if tokenize:
        from nltk.tokenize import word_tokenize
    sentences = [s.split() if not tokenize else word_tokenize(s)
                 for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict

if __name__ == '__main__':
    output_path = "/home/zxj/Data/sent_embedding_data/word_output/capital_world_words_embeddings.h5"
    with h5py.File(output_path, "r") as out_file:
        print(out_file.keys())
