from __future__ import print_function, unicode_literals

import io
import logging
import os
import sys

import torch

from gensen.gensen import GenSenSingle
from infer_sent.models import InferSent
from skip_thoughts import encoder_manager, configuration


def read_file(file_path, encoding="utf-8", preprocess=lambda x: x):
    try:
        with io.open(file_path, encoding=encoding) as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def restore_infer_sent(model_path, dict_path, version, use_cuda=False, batch_size=64):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    params_model = {'bsize': batch_size, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infer_sent = InferSent(params_model)
    infer_sent.load_state_dict(torch.load(model_path, map_location=device))

    infer_sent.set_w2v_path(dict_path)

    return infer_sent


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


def output_iterator(file_path, output_list, process=lambda x: x):
    try:
        with io.open(file_path, mode="w+", encoding="utf-8") as file:
            for line in output_list:
                file.write(process(line) + "\n")
    except IOError as error:
        logging.error("Failed to open file {0}".format(error))
        sys.exit(1)
