import os
from itertools import product
import main
import argparse

parser = argparse.ArgumentParser(description='LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--condition_runs', type=int, default=5,
                    help='Runs per condition')
parser.add_argument('--output_dir', type=str, default='./results/grid_search',
                    help='path to save results, including summary CSV and model checkpoint')
parser.add_argument('--summary_filename', type=str, default='summary.txt',
                    help='path to save summary CSV, within results directory')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use CUDA')
parser.add_argument('--start_condition', type=int, default=0,
                    help='Which condition number to start at')
gs_args = parser.parse_args()

condition_runs = gs_args.condition_runs
results_dir = gs_args.output_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

grid_args = {
    'data': gs_args.data,
    'model': ['LSTM', 'GRU'],
    'epochs': 40,
    'emsize': [500, 2000],
    'nhid': [32, 64],
    'nlayers': [1, 2],
    'ignore_speaker': [False, True]
    'lr': 20,
    'clip': 5,
    'cuda': gs_args.cuda,
    'summary': os.path.join(results_dir, gs_args.summary_filename)
}

def conditions(grid_args):
    """ Function that generates a list of dictionaries, with each dictionary representing the arguments of a condition. """

    arg_keys = list(grid_args.keys())
    # listify things that aren't lists
    arg_values = tuple([x if isinstance(x,list) else [x] for x in grid_args.values()])
    arg_combos = list(product(*arg_values))

    conditions = [dict(zip(arg_keys, x)) for x in arg_combos]

    return conditions

# generate and run conditions
for c, condition in enumerate(conditions(grid_args)):
    condition_index = c + gs_args.start_condition
    for run in range(condition_runs):
        condition['seed'] = run # use a randomly-generated seed instead?
        condition['run'] = run
        condition['condition'] = condition_index
        condition['save'] = os.path.join(results_dir, str(condition_index) + '-' + str(run) + '.pt')

        # could potentially parallelize here to run multiple runs at the same time?
        main.main(condition)
