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

(still need to figure out how to do)

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
