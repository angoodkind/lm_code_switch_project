import os
import torch
import csv
import re

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.speakers = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.csv'))
        self.valid = self.tokenize(os.path.join(path, 'valid.csv'))
        self.test = self.tokenize(os.path.join(path, 'test.csv'))

    def tokenize(self, path):
        """Get the data into word-target pairs for each sentence."""
        assert os.path.exists(path)
        # Add words to the dictionary
        convo_lengths = {}
        with open(path, 'r') as f:
            r = csv.reader(f)
            for line in r:
                words = line[2].split() + ['<eos>']
                # keep track of number of words in conversation
                if not line[0] in convo_lengths:
                    convo_lengths[line[0]] = []
                convo_lengths[line[0]].append(len(words))
                for word in words:
                    self.dictionary.add_word(word)
                self.speakers.add_word(line[1]) # build up speaker dictionary too

        convos = {k : [] for k in convo_lengths.keys()}

        chinese_chars = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)

        # Tokenize file content
        with open(path, 'r') as f:
            r = csv.reader(f)
            for line in r:
                words = line[2].split() + ['<eos>']
                speaker_id = self.speakers.word2idx[line[1]]
                enc_words = [self.dictionary.word2idx[word] for word in words]

                # input_tensor = torch.zeros(len(words), 1, len(self.speakers) + len(self.dictionary))
                input_sentence = torch.autograd.Variable(torch.LongTensor(enc_words))
                # input_sentence = torch.autograd.Variable(torch.LongTensor(4,5))
                output_classes = torch.LongTensor(len(enc_words), 2).zero_()
                # output_tensor = torch.zeros(len(words), 1, 2)
                for w, word_idx in enumerate(enc_words):
                #     input_tensor[w][0][speaker_id] = 1 # encode speaker id
                #     input_tensor[w][0][len(self.speakers) + word_idx] # encode word identity
                    # check for code-switches
                    if words[w] == '<eos>' or re.search(chinese_chars, words[w+1]) == re.search(chinese_chars, words[w]):
                        output_classes[w, 0] = 1
                    else: # code switch detected!
                        output_classes[w, 1] = 1

                # convos[line[0]].append((torch.LongTensor(input_tensor), torch.LongTensor(output_tensor)))
                convos[line[0]].append((input_sentence, torch.autograd.Variable(output_classes)))

        return convos
