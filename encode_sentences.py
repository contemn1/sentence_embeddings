from __future__ import absolute_import, print_function, unicode_literals

import argparse
import io
import json
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

from io_util import restore_infer_sent, read_file, restore_skipthought, restore_gen_sen
from quick_thought.configuration import model_config as quickthought_config
from quick_thought.encoder_manager import EncoderManager

NAME_WITH_SUFFIX = re.compile(r"\.[a-zA-Z0-9]+$")


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")

    parser.add_argument("--input-dir", type=str, metavar="N",
                        default="/home/zxj/Documents",
                        help="path of data directory")

    parser.add_argument("--input-file-name", type=str,
                        default="test_data.txt",
                        help="name of input file")

    parser.add_argument("--batch-size", type=int,
                        default=64,
                        help="path of glove file")

    parser.add_argument("--use-cuda", type=bool,
                        default=True,
                        help="path of glove file")

    parser.add_argument("--word2vec-path", type=str,
                        default="/home/zxj/Data/models/crawl-300d-2M.vec",
                        help="path of glove file")

    parser.add_argument("--infer-sent-model-path", type=str,
                        default="/home/zxj/Data/models/infersent{0}.pkl",
                        help="path of InferSent models")

    parser.add_argument("--infer-sent-version", type=int,
                        default=2,
                        help="path of InferSent models")

    parser.add_argument("--gensen-model-path", type=str,
                        default="/home/zxj/Data/gensen_models",
                        help="directory of general purpose sentence encoder model")

    parser.add_argument("--gensen-prefix", type=str,
                        default="nli_large_bothskip",
                        help="prefix of gen sen model")

    parser.add_argument("--gensen_embedding", type=str,
                        default="/media/zxj/sent_embedding_data/glove.840B.300d.h5",
                        help="prefix of pretrained gen sen word embedding")

    parser.add_argument("--skipthought-path", type=str, metavar="N",
                        default="/media/zxj/sent_embedding_data/skip_thoughts_uni_2017_02_02",
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
                        default="/media/zxj")

    parser.add_argument("--output-dir", type=str,
                        default="/home/zxj/Data")

    return parser


def infersent_encoder(model_path, word2vec_path, batch_size=128,
                      version=2, use_cuda=True):
    def infersent_embedding(sentences):
        model = restore_infer_sent(model_path, word2vec_path, version, use_cuda)  # InferSent
        model.build_vocab(sentences, tokenize=True)
        if use_cuda:
            model = model.cuda()

        return model.encode(sentences, bsize=batch_size, tokenize=True,
                            verbose=False)

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
        quick_thought_embedding = quick_thoughts_manager.encode(sentences, batch_size=batch_size)
        return quick_thought_embedding


def main(args):
    input_path = os.path.join(args.input_dir, args.input_file_name)
    sentences = list(read_file(input_path, preprocess=lambda x: x.strip()))

    args.infer_sent_model_path = args.infer_sent_model_path.format(args.infer_sent_version)

    universal_sent_encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

    infersent_embedding = infersent_encoder(args.infer_sent_model_path,
                                            args.word2vec_path,
                                            args.batch_size, args.use_cuda)(sentences)
    name_without_suffix = NAME_WITH_SUFFIX.sub("", args.input_file_name)
    infersent_embedding_path = os.path.join(args.output_dir, "{0}_infersent".format(name_without_suffix))
    np.save(infersent_embedding_path, infersent_embedding)

    skip_thought_encoder = restore_skipthought(args.skipthought_path, args.skipthought_model_name,
                                               args.skipthought_embeddings, args.skipthought_vocab_name)

    skip_thought_embedding = skip_thought_encoder.encode(sentences, batch_size=args.batch_size)

    skip_thought_embedding_path = os.path.join(args.output_dir, "{0}_skip_thought".format(name_without_suffix))
    np.save(skip_thought_embedding_path, skip_thought_embedding)

    quick_thought_embedding = get_quick_thought_embedding(args.quick_thought_config_path, sentences,
                                                          result_path=args.quick_thought_result_path,
                                                          batch_size=args.batch_size)

    quick_thought_embedding_path = os.path.join(args.output_dir, "{0}_quick_thought".format(name_without_suffix))
    np.save(quick_thought_embedding_path, quick_thought_embedding)

    gen_sen_encoder = restore_gen_sen(args.gensen_model_path, args.gensen_prefix, args.gensen_embedding)
    gen_sen_embedding = gen_sen_encoder.get_representation(sentences, tokenize=True)

    gen_sen_embedding_path = os.path.join(args.output_dir, "{0}_gen_sen".format(name_without_suffix))
    np.save(gen_sen_embedding_path, gen_sen_embedding)

    universal_sentence_embedding = get_universal_sent_embedding(universal_sent_encoder_url, sentences)
    universal_sentence_embedding_path = os.path.join(args.output_dir,
                                                     "{0}_universal_sentence".format(name_without_suffix))
    np.save(universal_sentence_embedding_path, universal_sentence_embedding)


if __name__ == '__main__':
    args = init_argument_parser().parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    main(args)