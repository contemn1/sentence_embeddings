# Sentence Embeddings
This project contains the source code of 5 sentence embedding methods: [InferSent](https://github.com/facebookresearch/InferSent),
[Skip-Thought Vector](https://github.com/tensorflow/models/tree/master/research/skip_thoughts), [Quick-Thought-Vectors](https://github.com/lajanugen/S2V),
[General Purpose Sentence Representation](https://github.com/Maluuba/gensen) and Universal Sentence Enoder("https://arxiv.org/pdf/1803.11175.pdf")
You can take a file consisting of multiple sentences as the input and get 5 different sentence embeddings file

## Dependencies
The code is written in python. Detailed dependencies despription could be found in enviroment.yml 

~~~~
conda env create -f environment.yml  
~~~~

You can use above bash script to create new environment

## Pretrained Models and Word Embeddings
All the pre-trained models and word embeddings could be found via this [link](https://drive.google.com/drive/folders/1ZtcBWNIZ8HW_J7DB8tW2LAq2AZlgHVIJ?usp=sharing).
Run
~~~~
python glove2h5.py    
~~~~
to generate glove embedding necessary for general purpose sentence representatin model

## Encoding Sentences
In this example, each line of the input file is a unique sentence
Just Run
~~~~
sh encode_sentences.sh  
~~~~
Don't forget to replace variables like INPUT_DIR, INPUT_FILE_NAME with actual file path in your environment.
