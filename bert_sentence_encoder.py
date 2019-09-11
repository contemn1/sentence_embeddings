import argparse
import os
import re

import h5py
import numpy as np
import torch
from pytorch_transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, BertModel, XLNetModel, RobertaModel
from torch.utils.data import DataLoader

from bert_dataset import BertDataset
from io_util import read_file

NAME_WITH_SUFFIX = re.compile(r"\.[a-zA-Z0-9]+$")


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")

    parser.add_argument("--input-path", type=str, metavar="N",
                        default="/home/zxj/Data/sentence_analogy_datasets/capital_country_pairs.txt",
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
    :type tokenizer: BertTokenizer
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
                    result_list.append(encoded_layers[index, seq_lengths[index] - 1])
                pooler_output = torch.stack(result_list).squeeze(1)
            else:
                pooler_output = res_tuple[1]

            average_pooling_batch = get_average_pooling(encoded_layers, masks)
            pool_result = pooler_output.cpu().numpy() if pool_result is None else np.vstack(
                (pool_result, pooler_output.cpu().numpy()))
            average_pooling_result = average_pooling_batch.cpu().numpy() if average_pooling_result is None else np.vstack(
                (average_pooling_result, average_pooling_batch.cpu().numpy()))

    return pool_result, average_pooling_result


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
    sentence_iterator = read_file(input_path, preprocess=lambda x: x.strip().split("\t")[-2:])
    sentence_list = [sent for arr in sentence_iterator for sent in arr]
    out_file = h5py.File(output_path, "r+")
    for model, tokenizer, model_names in models:
        for name in model_names:
            dataset_name = "-".join([ele.upper() for ele in name.split("-")[:2]])
            bert_tokenizer = tokenizer.from_pretrained(name)
            model = model.from_pretrained(name)
            bert_dataset = BertDataset(sentence_list, tokenizer=bert_tokenizer,
                                       with_special_tokens=args.with_special_tokens)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                                     collate_fn=dataset.collate_fn_one2one,
                                     pin_memory=args.use_cuda)

            cls_pooled_embeddings, average_pooled_embeddings = get_embedding_from_bert(model, bert_dataset
                                                                                       )
            try:
                out_file[dataset_name + "-CLS"] = cls_pooled_embeddings
                out_file[dataset_name + "-AVG"] = average_pooled_embeddings
            except RuntimeError as err:
                del out_file[dataset_name + "-CLS"]
                del out_file[dataset_name + "-AVG"]
                out_file[dataset_name + "-CLS"] = cls_pooled_embeddings
                out_file[dataset_name + "-AVG"] = average_pooled_embeddings

    out_file.close()


if __name__ == '__main__':
    args = init_argument_parser().parse_args()
    main(args)
