import re

import torch
import numpy as np
from typing import Union, List
from collections import Counter
from itertools import islice, tee
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class Batcher:
    def __init__(self, data: str, model_type: str = 'cbow', min_count: str = 5):
        if model_type not in {'cbow', 'skip-gram'}:
            raise ValueError('model_type should be "cbow" or "skip-gram"')

        self.model_type = model_type
        self.min_count = min_count
        self.word_count: dict = None
        self.word2index: dict = None
        self.index2word: dict = None
        self.unk_token: str = 'UNK'

        self.word_probs: np.ndarray = None
        self.word_indexes: np.ndarray = None

        self.data_path = data

        self.build_vocab()

    def preprocess(self, sentence):
        sentence = re.sub(r'\s+', ' ', sentence)
        tokens = word_tokenize(sentence.lower())
        # replace all numbers with [NUM]
        tokens = ['[NUM]' if re.match(r'[\d,.:]', token) else token for token in tokens]
        return tokens

    def _iter_lines(self):
        with open(self.data_path) as f:
            for line in f.readlines():
                yield self.preprocess(line)

    @property
    def vocab_size(self) -> int:
        return len(self.word2index.keys())

    def build_vocab(self):
        word_count = Counter()
        for words in tqdm(self._iter_lines(), desc='Making a dictionary of words', position=0):
            word_count.update(words)

        self.word_count = {}
        unk_count = 0
        for word, count in word_count.items():
            if word_count[word] > self.min_count:
                self.word_count[word] = count
            else:
                unk_count += count
        self.word_count[self.unk_token] = unk_count

        self.index2word = dict(enumerate(self.word_count.keys()))
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

        self.word_probs = np.fromiter(self.word_count.values(), dtype=np.float)
        self.word_probs = self.word_probs ** (3 / 4)
        self.word_probs /= self.word_probs.sum()
        self.word_indexes = np.fromiter(self.index2word.keys(), dtype=np.int)

    def generate_batches(self, batch_size: int = 1, window_size: int = 5, num_negative_samples: int = 0):
        x_positive, y, x_negative = np.zeros([batch_size, 2 * window_size], dtype=np.int), \
                                    np.zeros(batch_size, dtype=np.int), None  # cbow setup
        if num_negative_samples > 0:
            x_negative = np.zeros([batch_size, num_negative_samples], dtype=np.int)

        if self.model_type == 'skip-gram':
            x_positive, y = y, x_positive

        row_index = 0
        for sentence in self._iter_lines():
            for example in self._sliding_window(sentence, window_size * 2 + 1):
                num_example = self._numericalize(example)
                if self.model_type == 'cbow':
                    x_positive[row_index] = num_example[:window_size] + num_example[window_size + 1:]
                    y[row_index] = num_example[window_size]
                else:
                    x_positive[row_index] = num_example[window_size]
                    y[row_index] = num_example[:window_size] + num_example[window_size + 1:]

                if num_negative_samples > 0:
                    x_negative[row_index] = self._sample_negative(num_negative_samples)
                    # TODO x_negative all zeros after first iteration
                row_index += 1

                # this code will drop last batch if there is not enough data to generate it
                if row_index == batch_size:
                    row_index = 0
                    batch = {'x_positive': torch.tensor(x_positive, dtype=torch.long),
                             'y': torch.tensor(y, dtype=torch.long)}
                    if num_negative_samples > 0:
                        negative_samples = self._sample_negative(num_negative_samples * batch_size)\
                                               .reshape(batch_size, num_negative_samples)
                        batch['x_negative'] = torch.tensor(negative_samples, dtype=torch.long)
                    yield batch

    def _sample_negative(self, num_samples: int = 10) -> np.ndarray:
        # For better performance sample negative examples only ones for batch
        return np.random.choice(self.word_indexes, num_samples, replace=len(self.word_indexes) > num_samples,
                                p=self.word_probs)

    def _numericalize(self, sentence):
        return [self.word2index[w] if w in self.word2index else self.word2index[self.unk_token] for w in sentence]

    def _sliding_window(self, iterable: list, window_size: int = 3):
        yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(iterable, window_size))])
