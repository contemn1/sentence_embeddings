import argparse
import os
import re

import h5py
import numpy as np
import torch
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, BertModel, XLNetModel, RobertaModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from bert_dataset import BertDataset
from io_util import read_file, read_relation_analogy, read_word_based_analogy

NAME_WITH_SUFFIX = re.compile(r"\.[a-zA-Z0-9]+$")


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")

    parser.add_argument("--input-path", type=str, metavar="N",
                        default="/home/zxj/Data/sentence_analogy_datasets/city_in_state_pairs.txt",
                        help="path of data directory")

    parser.add_argument("--batch-size", type=int,
                        default=64,
                        help="path of glove file")

    parser.add_argument("--use-cuda", action='store_true',
                        default=False,
                        help="path of glove file")

    parser.add_argument("--with-special-tokens", action='store_true',
                        default=False,
                        help="path of glove file")

    parser.add_argument("--output-dir", type=str,
                        default="/home/zxj/Data/sent_embedding_data/output")

    return parser


def get_embedding_from_bert(model, data_loader):
    """
    :type model: BertModel
    :type data_loader: DataLoader
    :return:
    """
    pool_result = None
    bert = model
    average_pooling_result = None

    for ids, masks in data_loader:
        if torch.cuda.is_available():
            ids = ids.cuda()
            masks = masks.cuda()
            bert = model.cuda()

        model.eval()
        with torch.no_grad():
            res_tuple = bert(ids, attention_mask=masks)
            encoded_layers = res_tuple[0]
            seq_lengths = masks.sum(dim=1, keepdim=True)
            if isinstance(model, XLNetModel):
                result_list = []
                for index in range(seq_lengths.shape[0]):
                    result_list.append(encoded_layers[index, seq_lengths[index] - 3])
                pooler_output = torch.stack(result_list).squeeze(1)
            else:
                pooler_output = res_tuple[1]

            average_pooling_batch = get_average_pooling(encoded_layers, masks)
            pool_result = pooler_output.cpu().numpy() if pool_result is None else np.vstack(
                (pool_result, pooler_output.cpu().numpy()))
            average_pooling_result = average_pooling_batch.cpu().numpy() if average_pooling_result is None else np.vstack(
                (average_pooling_result, average_pooling_batch.cpu().numpy()))

    return pool_result, average_pooling_result


def get_word_embedding_from_bert(model, data_loader):
    """
    :type model: BertModel
    :type data_loader: DataLoader
    :return:
    """
    bert = model
    average_pooling_result = None

    for ids, masks in data_loader:
        if torch.cuda.is_available():
            ids = ids.cuda()
            masks = masks.cuda()
            bert = model.cuda()

        model.eval()
        with torch.no_grad():
            res_tuple = bert(ids, attention_mask=masks)
            encoded_layers = res_tuple[0]
            average_pooling_batch = get_average_pooling(encoded_layers, masks)
            average_pooling_result = average_pooling_batch.cpu().numpy() if average_pooling_result is None else np.vstack(
                (average_pooling_result, average_pooling_batch.cpu().numpy()))

    return average_pooling_result


def get_average_pooling(embeddings, masks):
    sum_embeddings = torch.bmm(masks.unsqueeze(1).float(), embeddings).squeeze(1)
    average_embeddings = sum_embeddings / torch.sqrt(masks.sum(dim=1, keepdim=True).float())
    return average_embeddings


def main(args):
    models = [(BertModel, BertTokenizer, ['bert-base-cased', 'bert-large-cased']),
              (XLNetModel, XLNetTokenizer, ['xlnet-base-cased', 'xlnet-large-cased']),
              (RobertaModel, RobertaTokenizer, ['roberta-base', 'roberta-large'])]

    input_path = args.input_path
    input_file_name = os.path.basename(input_path)
    name_without_suffix = NAME_WITH_SUFFIX.sub("", input_file_name)
    output_path = os.path.join(args.output_dir, "{0}_embeddings.h5".format(name_without_suffix))
    sentence_list = read_relation_analogy(args)
    out_file = h5py.File(output_path, "r+")
    for model, tokenizer, model_names in models:
        for name in model_names:
            dataset_name = "-".join([ele.upper() for ele in name.split("-")[:2]])
            bert_tokenizer = tokenizer.from_pretrained(name)
            model = model.from_pretrained(name)
            bert_dataset = BertDataset(sentence_list, tokenizer=bert_tokenizer,
                                       with_special_tokens=args.with_special_tokens)
            data_loader = DataLoader(bert_dataset, batch_size=args.batch_size, num_workers=0,
                                     collate_fn=bert_dataset.collate_fn_one2one,
                                     pin_memory=args.use_cuda)

            cls_pooled_embeddings, average_pooled_embeddings = get_embedding_from_bert(model, data_loader)

            cls_key, avg_key = dataset_name + "-CLS", dataset_name + "-AVG"
            if dataset_name in out_file.keys():
                del out_file[dataset_name]
            if cls_key in out_file.keys():
                del out_file[cls_key]
            if avg_key in out_file.keys():
                del out_file[avg_key]

            out_file[cls_key] = cls_pooled_embeddings
            out_file[avg_key] = average_pooled_embeddings

    out_file.close()


def encode_words():
    args = init_argument_parser().parse_args()
    models = [(BertModel, BertTokenizer, ['bert-base-cased', 'bert-large-cased']),
              (XLNetModel, XLNetTokenizer, ['xlnet-base-cased', 'xlnet-large-cased']),
              (RobertaModel, RobertaTokenizer, ['roberta-base', 'roberta-large'])]
    input_path = args.input_path
    input_file_name = os.path.basename(input_path)
    name_without_suffix = NAME_WITH_SUFFIX.sub("", input_file_name)
    category_name = re.sub(r"_words_embeddings.h5", "", name_without_suffix)
    sentence_iterator = read_file(input_path, preprocess=lambda x: x.strip().split("\t")[-2:])
    sentence_list = [sent for arr in sentence_iterator for sent in arr]
    for model, tokenizer, model_names in models:
        bert_tokenizer = tokenizer.from_pretrained(model_names[0])
        print(category_name + ": " + model_names[0])
        word_count = 0
        for ele in sentence_list:
            result_list = bert_tokenizer.tokenize(ele)
            if len(result_list) > 1:
                print(ele + "\t" + " ".join(result_list))
                word_count += 1
        print(float(word_count) * 2 / len(sentence_list))


def add_sbert_encoding():
    input_path = args.input_path
    input_file_name = os.path.basename(input_path)
    name_without_suffix = NAME_WITH_SUFFIX.sub("", input_file_name)
    output_path = os.path.join(args.output_dir, "{0}_embeddings.h5".format(name_without_suffix))
    sentence_iterator = read_file(input_path, preprocess=lambda x: x.strip().split("\t")[-2:])
    sentence_list = [sent for arr in sentence_iterator for sent in arr]
    new_name_dict = {"SBERT-Base-AVG": "bert-base-nli-mean-tokens",
                     "SBERT-Large-AVG": "bert-large-nli-mean-tokens",
                     "SBERT-Base-CLS": "bert-base-nli-cls-token",
                     "SBERT-Large-CLS": "bert-large-nli-cls-token",
                     "SRoBERTa-Base-AVG": "roberta-base-nli-mean-tokens",
                     "SRoBERTa-Large-AVG": "roberta-large-nli-mean-tokens"}
    out_file = h5py.File(output_path, "r+")
    for output_name, model_name in new_name_dict.items():
        model = SentenceTransformer(model_name)
        embedding_list = model.encode(sentence_list, batch_size=32)
        embeddings = np.vstack(embedding_list)
        if output_name in out_file.keys():
            del out_file[output_name]

        out_file[output_name] = embeddings


if __name__ == '__main__':
    args = init_argument_parser().parse_args()
    main(args)
    # out_file.close()
