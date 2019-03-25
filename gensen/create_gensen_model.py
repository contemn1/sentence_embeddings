"""Creates a GenSen model from a MultiSeq2Seq model."""
import os
import pickle
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--trained_model_folder",
    help="Path to the folder containing a saved model",
    required=True,
    type=str
)
parser.add_argument(
    "-s",
    "--save_folder",
    help="Path to save the encoder",
    required=True,
    type=str
)
parser.add_argument(
    "-n",
    "--save_name",
    help="Name of the model",
    required=True,
    type=str
)
args = parser.parse_args()

model = torch.load(
    open(os.path.join(args.trained_model_folder, 'best_model.model'))
)

for item in model.keys():
    if (
        not item.startswith('module.encoder') and
        not item.startswith('module.src_embedding')
    ):
        del model[item]

for item in model.keys():
    model[item.replace('module.', '')] = model[item]

for item in model.keys():
    if item.startswith('module.'):
        del model[item]

torch.save(
    model,
    os.path.join(args.save_folder, '%s.model' % (args.save_name))
)

model_vocab = pickle.load(
    open(os.path.join(args.trained_model_folder, 'src_vocab.pkl'))
)
pickle.dump(
    model_vocab,
    open(
        os.path.join(args.save_folder, '%s_vocab.pkl' % (args.save_name)), 'wb'
    )
)
