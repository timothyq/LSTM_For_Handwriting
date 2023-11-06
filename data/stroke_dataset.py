import torch
import numpy as np
from torch.utils.data import Dataset


class Stroke_dataset(Dataset):
    def __init__(self, data_path="data/strokes-py3.npy", sentences_path="data/sentences.txt", train=True):

        data = np.load(data_path, allow_pickle=True)
        self.data = [torch.tensor(item) for item in data if isinstance(
            item, (list, np.ndarray))]

        with open(sentences_path, "r") as f:
            raw_sentences = f.readlines()

        # Build a character-level vocabulary
        self.vocab = self.build_vocab(raw_sentences)

        # One-hot encode each sentence
        self.sentences = [self.one_hot_encode(
            sentence) for sentence in raw_sentences]

    def build_vocab(self, sentences):
        vocab = set()
        for sentence in sentences:
            for char in sentence:
                vocab.add(char)
        return sorted(list(vocab))

    def one_hot_encode(self, sentence):
        one_hot = torch.zeros(len(sentence), len(self.vocab))
        for idx, char in enumerate(sentence):
            char_idx = self.vocab.index(char)
            one_hot[idx, char_idx] = 1
        return one_hot  # (len(sentence), len(self.vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.sentences[index]

    # def __getitem__(self, index):
    #     # padding version of the dataset
    #     data_item = self.data[index]
    #     padding_length = 992 - len(data_item)
    #     if padding_length > 0:
    #         data_item = F.pad(data_item, (0, 0, 0, padding_length), 'constant', 0)

    #     condition_seq = self.sentences[index]
    #     # Assuming condition_seq is a tensor.
    #     if padding_length > 0:
    #         condition_seq = F.pad(condition_seq, (0, 0, 0, padding_length), 'constant', 0)

    #     return data_item, condition_seq
