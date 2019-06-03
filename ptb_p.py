import codecs
import collections
from operator import itemgetter
import tensorflow as tf

RAW_DATA = r'D:\data\simple-examples\data\ptb.train.txt'
VOCAB_OUPUT = 'ptb.vocab'
OUTPUT_DATA = 'ptb.train'

TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEPS = 35

counter = collections.Counter()
print(counter[1])
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ['<eos>'] + sorted_words

with codecs.open(VOCAB_OUPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')

import sys

word_to_id = {k: v for (k, v) in zip(sorted_words, range(len(sorted_words)))}


def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


raw_data = codecs.open(RAW_DATA, 'r', 'utf-8')
with codecs.open(OUTPUT_DATA, 'w', 'utf-8') as file:
    for line in raw_data:
        words = line.strip().split() + ['<eos>']
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        file.write(out_line)
raw_data.close()


def read_data(file_path):
    with open(file_path, 'r') as file:
        id_string = ' '.join([line.strip() for line in file.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_steps):
    num_batches = (len(id_list) - 1) // (batch_size * num_steps)
    data = np.array(id_list[:num_batches * batch_size * num_steps])
    data = np.reshape(data, [batch_size, num_batches * num_steps])
    data_batchs = np.split(data, num_batches, axis=1)

    label = np.array(id_list[1:num_batches * batch_size * num_steps+1])
    label = np.reshape(label, [batch_size, num_steps * num_batches])
    label_batchs = np.split(label, num_batches, axis=1)

    return list(zip(data_batchs, label_batchs))


def main():
    train_batch = make_batches(read_data(OUTPUT_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)


import numpy as np

if __name__ == '__main__':
    main()
