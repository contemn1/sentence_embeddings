import numpy as np
import torch
from pytorch_transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, word_sequence, tokenizer, with_special_tokens=True,
                 max_length=128):
        """
        :type word_sequence: list
        :type tokenizer: PreTrainedTokenizer
        :type with_special_token: bool
        :type max_length: int
        :param word_sequence: List of sentences or sentence pairs
        :param tokenizer: used to tokenize sentences and map words to ids
        :param with_special_token: if set to True, the sequences will be encoded with the special tokens relative to their model
        :param max_length:  max number of words in a sequence
        """

        self.raw_texts = word_sequence
        self.tokenizer = tokenizer  # type: PreTrainedTokenizer
        self.max_length = max_length
        self.with_special_tokens = with_special_tokens

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, index):
        result = self.tokenizer.encode(self.raw_texts[index], add_special_tokens=self.with_special_tokens)[
               :self.max_length]
        return result

    def collate_fn_one2one(self, bert_ids):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        sequence_lengths = np.array([len(ele) for ele in bert_ids])
        padded_batch_ids = pad(bert_ids, sequence_lengths,
                               self.tokenizer.pad_token_id)  # type: torch.Tensor
        input_masks = (padded_batch_ids != self.tokenizer.pad_token_id).to(torch.uint8)
        return padded_batch_ids, input_masks


def pad(sequence_raw, sequence_length, pad_id):
    def pad_per_line(index_list, max_length):
        return np.concatenate(
            (index_list, [pad_id] * (max_length - len(index_list))))

    max_seq_length = np.max(sequence_length)
    padded_sequence = np.array(
        [pad_per_line(x_, max_seq_length) for x_ in sequence_raw],
        dtype=np.int64)

    padded_sequence = torch.from_numpy(padded_sequence)

    return padded_sequence
