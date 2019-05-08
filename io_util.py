from __future__ import print_function, unicode_literals

import io
import logging
import os
import sys

import torch



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
