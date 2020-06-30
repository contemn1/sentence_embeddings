from __future__ import absolute_import, print_function, unicode_literals

import argparse
import io
import json
import os
import re
from functools import partial

import h5py
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
from nltk.tokenize import word_tokenize

from gensen.gensen import GenSenSingle
from infer_sent.models import InferSent
from io_util import read_relation_analogy
from quick_thought.configuration import model_config as quickthought_config
from quick_thought.encoder_manager import EncoderManager
from skip_thoughts import encoder_manager, configuration
from weighting_functions import get_dct_coefficient

NAME_WITH_SUFFIX = re.compile(r"\.[a-zA-Z0-9]+$")


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")

    parser.add_argument("--input-path", type=str, metavar="N",
                        default="/home/zxj/Data/sentence_analogy_datasets/dict/capital_world_words.txt",
                        help="path of data directory")

    parser.add_argument("--model-dir", type=str, metavar="N",
                        default="/home/zxj/Data/sent_embedding_data",
                        help="path of model root directory")

    parser.add_argument("--batch-size", type=int,
                        default=64,
                        help="path of glove file")

    parser.add_argument("--use-cuda", type=bool,
                        default=True,
                        help="path of glove file")

    parser.add_argument("--infer-sent-version", type=int,
                        nargs="+",
                        default=[1, 2],
                        help="path of InferSent models")

    parser.add_argument("--gensen-model-path", type=str,
                        default="/home/zxj/Data/sent_embedding_data/gensen_models",
                        help="directory of general purpose sentence encoder model")

    parser.add_argument("--gensen-prefix", type=str,
                        default="nli_large_bothskip",
                        help="prefix of gen sen model")

    parser.add_argument("--gensen_embedding", type=str,
                        default="/home/zxj/Data/sent_embedding_data/glove.840B.300d.h5",
                        help="prefix of pretrained gen sen word embedding")

    parser.add_argument("--skipthought-path", type=str, metavar="N",
                        default="/home/zxj/Data/sent_embedding_data/skip_thoughts_uni_2017_02_02",
                        help="path of pre-trained skip-thought vectors model")

    parser.add_argument("--skipthought-model-name", type=str,
                        default="model.ckpt-501424",
                        help="name of pre-trained skip-thought vectors model")

    parser.add_argument("--skipthought-embeddings", type=str,
                        default="embeddings.npy",
                        help="name of pre-trained skip-thought word embeddings model")

    parser.add_argument("--skipthought-vocab-name", type=str,
                        default="vocab.txt",
                        help="name of pre-trained skip-thought vectors vocabulary")

    parser.add_argument("--quick-thought-config-path", type=str,
                        default="/home/zxj/PycharmProjects/sentence_embeddings/model_configs/MC-UMBC/eval.json",
                        help="path of config file")

    parser.add_argument("--quick-thought-result-path", type=str,
                        default="/home/zxj/Data/sent_embedding_data")

    parser.add_argument("--output-dir", type=str,
                        default="/home/zxj/Data/sent_embedding_data")

    return parser


def restore_infer_sent(model_path, dict_path, version, use_cuda=False, batch_size=64):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    params_model = {'bsize': batch_size, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infer_sent = InferSent(params_model)
    infer_sent.load_state_dict(torch.load(model_path, map_location=device))
    infer_sent.set_w2v_path(dict_path)

    return infer_sent


def pad_array(array, max_length):
    seq_length, dimension = array.shape
    pad_length = max_length - seq_length
    if pad_length > 0:
        return np.concatenate((array, np.zeros((pad_length, dimension))))
    else:
        return array


def get_dct_embeddings(k, word_embeddings):
    assert len(word_embeddings.shape) == 2
    num_words, dimension = word_embeddings.shape
    dct_coefficients = get_dct_coefficient(k, num_words)
    return np.expand_dims(dct_coefficients.dot(word_embeddings), axis=0)


def weighted_word_embedding_generator(embedding_dict, tokenize=True, weighting_function=lambda x: np.mean(x, axis=0)):
    def get_embedding(sentences):
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        word_embeddings = [np.vstack([embedding_dict[word] for word in sent if word in embedding_dict])
                           for sent in tokenized_sentences]
        sentence_embeddings = [weighting_function(array) for array in word_embeddings]
        return np.vstack(sentence_embeddings)

    return get_embedding


def restore_gen_sen(model_folder, filename_prefix, pretrained_emb, use_cuda=True):
    """

    :type model_folder: str
    :type filename_prefix: str
    :type pretrained_emb: str
    :type use_cuda: bool
    :rtype: GenSenSingle
    :return:
    """
    gensen_1 = GenSenSingle(
        model_folder=model_folder,
        filename_prefix=filename_prefix,
        pretrained_emb=pretrained_emb,
        cuda=use_cuda
    )
    return gensen_1


def create_gensen_list(sentence_iterator, word2id):
    gensen_iterator = (sent for ele in sentence_iterator for sent in [(ele[0], ele[2]), (ele[1], ele[3])])
    sentence_list = []
    for word, sent in gensen_iterator:  # type: str
        sent = sent.lower()
        if word.isupper() and word.lower() not in word2id:
            sent = re.sub(r"{0}".format(word.lower()), word, sent)
        sentence_list.append(sent)
    return sentence_list


def restore_skipthought(model_dir, model_name, skipthought_embedding, skipthought_vocab):
    """
    :rtype: encoder_manager.EncoderManager()
    :return:
    """
    check_point_path = os.path.join(model_dir, model_name)
    skip_thought_embedding_matrix = os.path.join(model_dir, skipthought_embedding)
    skip_thought_vocab = os.path.join(model_dir, skipthought_vocab)

    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                       vocabulary_file=skip_thought_vocab,
                       embedding_matrix_file=skip_thought_embedding_matrix,
                       checkpoint_path=check_point_path)
    return encoder


def infersent_encoder(model_path, word2vec_path, batch_size=64,
                      version=2, use_cuda=True):
    def infersent_embedding(sentences):
        model = restore_infer_sent(model_path, word2vec_path, version, use_cuda)  # InferSent
        model.build_vocab(sentences, tokenize=True)
        if use_cuda:
            model = model.cuda()

        return model

    return infersent_embedding


def get_universal_sent_embedding(module_url, sentence_list):
    embed = hub.Module(module_url)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(sentence_list))

    return message_embeddings


def get_quick_thought_embedding(config_path, sentences, glove_path=None, result_path=None, batch_size=64):
    with io.open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)[-1]
        model_config = quickthought_config(config, "encode", glove_path=glove_path, result_path=result_path)
        quick_thoughts_manager = EncoderManager()
        quick_thoughts_manager.load_model(model_config)
        quick_thought_embedding = quick_thoughts_manager.encode(sentences, batch_size=batch_size, use_norm=False)
        return quick_thought_embedding


def get_gensen_embedding(encoder, sentence_list, batch_size, tokenize=True):
    """
    :type encoder: GenSenSingle
    :type sentence_list: List[str]
    :type batch_size: int
    :param encoder:
    :param sentence_list:
    :param batch_size:
    :return:
    """

    def tokenize_func(x):
        return nltk.word_tokenize(x) if tokenize else x.split()

    gen_sen_embedding_list = []
    task_vocab = list(set((word.encode("utf-8") for sent in sentence_list for word in tokenize_func(sent))))
    encoder.vocab_expansion(task_vocab)

    for index in range(0, len(sentence_list), batch_size):
        sentences_per_batch = sentence_list[index: index + batch_size]
        _, gen_sen_embedding_per_batch = encoder.get_representation(sentences_per_batch, tokenize=tokenize)
        gen_sen_embedding_list.append(gen_sen_embedding_per_batch)

    return np.vstack(gen_sen_embedding_list)


def get_output_path(input_path, output_dir, template='{0}_embeddings.h5'):
    input_file_name = os.path.basename(input_path)
    name_without_suffix = NAME_WITH_SUFFIX.sub("", input_file_name)
    return os.path.join(output_dir, template.format(name_without_suffix))


def main(args, sentence_list):
    input_path = args.input_path
    output_path = get_output_path(input_path, args.output_dir)
    out_file = h5py.File(output_path, "w")
    sentence_list_output = np.array([sent.encode("utf-8") for sent in sentence_list])

    infersent_dir = os.path.join(args.model_dir, "infersent")
    infersent_dict_name = {1: "glove.840B.300d.txt", 2: "crawl-300d-2M.vec"}

    for version in args.infer_sent_version:
        infersent_model_path = os.path.join(infersent_dir, "infersent{0}.pkl".format(version))
        infersent_dict_path = os.path.join(infersent_dir, infersent_dict_name[version])
        infersent_model = infersent_encoder(infersent_model_path,
                                            infersent_dict_path,
                                            args.batch_size,
                                            version=version,
                                            use_cuda=args.use_cuda)(sentence_list)
        infersent_embedding = infersent_model.encode(sentence_list, bsize=args.batch_size, tokenize=True,
                                                     verbose=False)
        out_file.create_dataset(name="InferSentV{0}".format(version), data=infersent_embedding)

    word_embedding_dict = infersent_model.word_vec
    add_dct_embedding_to_result(sentence_list, word_embedding_dict, out_file)
    add_glove_embedding_to_result(sentence_list, word_embedding_dict, out_file)
    del word_embedding_dict
    del infersent_model

    out_file.create_dataset(name="Sentences", data=sentence_list_output)

    universal_sent_encoder_d_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    universal_sent_encoder_t_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    skip_thought_encoder = restore_skipthought(args.skipthought_path, args.skipthought_model_name,
                                               args.skipthought_embeddings, args.skipthought_vocab_name)

    skip_thought_embedding = skip_thought_encoder.encode(sentence_list, batch_size=args.batch_size, use_norm=False)

    out_file.create_dataset(name="SkipThought", data=skip_thought_embedding)

    universal_sentence_embedding_dan = get_universal_sent_embedding(universal_sent_encoder_d_url, sentence_list)
    universal_sentence_embedding_t = get_universal_sent_embedding(universal_sent_encoder_t_url, sentence_list)
    out_file.create_dataset(name="UniversalSentenceDAN", data=universal_sentence_embedding_dan)
    out_file.create_dataset(name="UniversalSentenceTransformer", data=universal_sentence_embedding_t)

    quick_thought_embedding = get_quick_thought_embedding(args.quick_thought_config_path, sentence_list,
                                                          result_path=args.quick_thought_result_path,
                                                          batch_size=args.batch_size)

    out_file.create_dataset(name="QuickThought", data=quick_thought_embedding)

    gen_sen_encoder = restore_gen_sen(args.gensen_model_path, args.gensen_prefix, args.gensen_embedding)
    gen_sen_embedding = get_gensen_embedding(gen_sen_encoder, sentence_list, args.batch_size)
    out_file.create_dataset(name="GenSen", data=gen_sen_embedding)

    out_file.close()


def add_dct_embedding_to_result(sentence_list, embedding_dict, out_file):
    weighting_func = partial(get_dct_embeddings, 7)
    embedding_generator = weighted_word_embedding_generator(embedding_dict, weighting_function=weighting_func)
    sentence_embeddings = embedding_generator(sentence_list)
    batch_size = sentence_embeddings.shape[0]
    for idx in range(1, 8):
        embeddings = sentence_embeddings[:, :idx].reshape(batch_size, -1)
        embedding_name = "$c^0$" if idx == 1 else "$c^{{0:{index}}}$".format(index=idx - 1)
        out_file.create_dataset(name=embedding_name, data=embeddings)


def add_glove_embedding_to_result(sentence_list, embedding_dict, out_file, dataset_name="GLOVE"):
    embedding_generator = weighted_word_embedding_generator(embedding_dict)
    sentence_embeddings = embedding_generator(sentence_list)
    if dataset_name not in out_file.keys():
        out_file.create_dataset(dataset_name, data=sentence_embeddings)


if __name__ == '__main__':
    args = init_argument_parser().parse_args()
    sentence_list = read_relation_analogy(args)
    main(args, sentence_list)