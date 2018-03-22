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

        convo_lengths = {}
        convos = {}

        # for checking whether a word is Chinese or English, later
        chinese_chars = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)

        with open(path, 'r') as f:
            r = csv.reader(f)
            convo_id = None
            for line in r:
                # reset convo
                if line[0] != convo_id:
                    # wrap up stuff from previous convo
                    if convo_id:
                        # create Variables from existing vectors
                        # input_complete: Variable of dimensionality 2 x (number of words in sentence)
                        input_complete = torch.autograd.Variable(torch.LongTensor([enc_words, speaker_tags]))
                        # output_classes
                        output_classes = torch.autograd.Variable(torch.LongTensor(output_classes))


                        convos[convo_id] = {
                            'input': input_complete,
                            'target': output_classes
                        }

                    convo_id = line[0]
                    enc_words = []
                    speaker_tags = []
                    output_classes = []
                    convo_lengths[convo_id] = []

                # split words into a list for processing
                words = line[2].split() + ['<eos>']

                # keep track of lengths
                convo_lengths[convo_id].append(len(words))

                # add words to dictionary
                for word in words:
                    self.dictionary.add_word(word)
                self.speakers.add_word(line[1]) # build up speaker dictionary too

                # encode both speaker tags and words
                enc_words += [self.dictionary.word2idx[word] for word in words]
                speaker_id = self.speakers.word2idx[line[1]]
                speaker_tags += [speaker_id for word in words]

                # figure out output classes
                # 0 = no codeswitch | 1 = codeswitch
                this_output = [1 for word in words]
                for w, word in enumerate(words):
                    # check for code-switches
                    if word == '<eos>' or re.search(chinese_chars, words[w+1]) == re.search(chinese_chars, word):
                        this_output[w] = 0 # no code-switch

                output_classes += this_output


        return convos
