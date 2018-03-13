# Predicting Code Switches in Conversation

A final project for EECS 496 (Language Modeling seminar) at Northwestern University.

### Acknowledgements

Much of this code was based on:

* the [word-level language model from PyTorch examples](https://github.com/pytorch/examples/tree/master/word_language_model), originally created by [Adam Lerer](https://github.com/adamlerer)
* the [sequence model code/tutorial](http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging) by [Robert Guthrie](https://github.com/rguthrie3).

## Dependencies

* Python 3
* [pytorch](http://pytorch.org/)

## Preprocessing the data

Run `src/preprocess.py` on your corpus, with the following arguments. Note that this preprocessing code is intended to work with the [SEAME corpus](https://catalog.ldc.upenn.edu/LDC2015S04), which cannot be published here due to copyright reasons.

- `--source_dir` = location of the data corpus (a directory with conversation transcripts)
- `--train_prop` = proportion of corpus to use as training. The rest will be split evenly between testing and validation sets
- `--output_dir` = where to save training/testing/validation splits, each as a CSV, where each line contains the following data:
    1. Conversation ID
    2. Speaker
    3. Utterance

Within a given conversation, all the lines are in order in the CSV.

## Running the experiment

To run a single set of parameters, simply run `src/main.py` with parameter settings:

- `--data` = location of the data corpus
- `--model` = type of recurrent net to use (RNN_TANH, RNN_RELU, LSTM, GRU)
- `--emsize` = size of word embeddings
- `--nhid` = number of hidden units per recurrent layer
- `--nlayers` = number of recurrent layers
- `--lr` = initial learning rate
- `--clip` = maximum value for gradient clipping
- `--epochs` = maximum epochs
- `--dropout` = amount of dropout applied to layers
- `--decay` = learning rate decay per epoch
- `--tied` = whether to tie the word embedding and softmx weights for faster training
- `--seed` = random seed (in grid search, this is set to the condition index)
- `--cuda` = use CUDA
- `--log-interval` = report interval
- `--save` = path where to save the final model

To run a grid search on several parameters, all you need to do is edit the variables in `src/grid_search.py`, and then run that file. Grid search has a couple of parameters configurable from the command line, too:

* `--data` = location of the data corpus
* `--condition_runs` = number of runs per condition (each run starts with a different random seed)
* `--output_dir` = path to save results, including summary CSV and model checkpoint
* `--summary_filename` = path to save summary CSV, within the results directory
* `--cuda` = use CUDA
