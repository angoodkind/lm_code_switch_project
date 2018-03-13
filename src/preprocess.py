import csv
import os
import argparse
import random
import json

# parse input arguments
parser = argparse.ArgumentParser(description='LSTM Language Model')
parser.add_argument('--source_dir', type=str, default='../seame/data/conversation/transcript/phaseII',
                    help='location of the data corpus')
parser.add_argument('--train_prop', type=float, default=.8,
                    help='Proportion of conversations to use as training corpus. (Validation and training sets will be of equal size, from the remaining data)')
parser.add_argument('--output_dir', type=str, default='./data/seame',
                    help='path to save processed data (train, valid, and test will be subdirectories)')
args = parser.parse_args()

###############################################
# 1. Retranscribe the files into combined files
###############################################

assert (os.path.exists(args.source_dir)), "Source directory does not exist"

# store filenames
file_list = [f for f in os.listdir(args.source_dir) if os.path.isfile(os.path.join(args.source_dir, f))]
assert (len(file_list) > 0), "Source directory does not contain any files"

# create output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# build data structure
convos = {}
for filename in file_list:
    convo_id = filename[0:2] + "_" + filename[10:12] # each session counted as separate convo
    subj = filename[4:6]

    # add to data structure
    if filename[0:2].isdigit() and subj.isdigit():
        if not convo_id in convos:
            convos[convo_id] = { subj : filename }
        else:
            convos[convo_id][subj] = filename

    # this line is not needed because there are no multipart transcripts here
    # part = filename[12:14]

for convo_id, convo_info in convos.items():
    convo_info['lines'] = {}
    speaker_files = {s: convo_info[s] for s in convo_info.keys() if s.isdigit()}
    for speaker, transcript in speaker_files.items():
        fn = os.path.join(args.source_dir, transcript)
        with open(fn, 'r') as transfile:
            reader = csv.reader(transfile, delimiter='\t')
            for row in reader:
                convo_info['lines'][int(row[1])] = [convo_id, speaker, row[4]]

# split into train, test, and validation sets
# random split
convo_ids = list(convos.keys())
random.shuffle(convo_ids)

# lengths of each split
n_train = int(args.train_prop * len(convo_ids))
n_valid = (len(convo_ids) - n_train) // 2
split = {}
split['train'] = {'ids' : convo_ids[0:n_train]}
split['valid'] = {'ids' : convo_ids[n_train:n_train + n_valid]}
split['test'] = {'ids' : convo_ids[n_train + n_valid:]}

for split_name, split_info in split.items():
    split[split_name]['n_lines'] = 0
    split[split_name]['n_words'] = 0
    split[split_name]['n_convos'] = len(split_info['ids'])
    writer = csv.writer(open(os.path.join(args.output_dir, split_name + '.csv'), 'w'))
    for convo_id in split_info['ids']:
        lines = [(time, line) for time, line in convos[convo_id]['lines'].items()]
        split[split_name]['n_lines'] += len(lines) # keep track of number of lines
        lines.sort(key=lambda tup: tup[0]) # sort lines by time
        for time, line in lines:
            split[split_name]['n_words'] += len(line[-1].split(" "))
            writer.writerow(line)

# generate a README
with open(os.path.join(args.output_dir, 'README'), 'w') as out:
    out.write(json.dumps(split, indent=4, sort_keys=True))
