def main(preset_args = False):

    # coding: utf-8
    import argparse
    import time
    import math
    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    import data
    import model
    import csv
    import os

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='./data/seame',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--summary', type=str, default='summary.txt',
                        help='path to save summary CSV')
    parser.add_argument('--condition', type=int, default=0,
                        help='Condition referenced in summary CSV')
    parser.add_argument('--run', type=int, default=0,
                        help='Run within condition')
    args, unknown = parser.parse_known_args()
    output_info = vars(args) # this is the variable to use for outputting checkpoints

    # maybe you've passed a dictionary into this function!
    if preset_args:
        vars(args).update(preset_args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)


    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    nspeakers = 0 # don't include speaker information for now
    model = model.RNNModel(args.model, ntokens, nspeakers, args.emsize, args.nhid, 2, args.nlayers, args.dropout)
    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden()
        for convo in data_source.values():
            for data, targets in convo:
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, 2)
                total_loss += len(data) * criterion(output_flat, targets).data
                hidden = repackage_hidden(hidden)
        return total_loss[0] / len(data_source)


    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden()
        loss_hist = []
        batch = 0
        n_batches = sum([len(c) for c in corpus.train.values()])
        for convo in corpus.train.values():
            for data, targets in convo:
                batch += 1

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = repackage_hidden(hidden)
                model.zero_grad()
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, 2), targets)
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)

                total_loss += loss.data
                loss_hist.append(loss.data[0])

                if batch % args.log_interval == 0 and batch > 0:
                    cur_loss = total_loss[0] / args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, n_batches, lr,
                        elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()

        return sum(loss_hist) / len(loss_hist)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(corpus.valid)

            output_info['epoch'] = epoch
            output_info['train_loss'] = train_loss
            output_info['val_loss'] = val_loss
            output_info['val_ppl'] = math.exp(val_loss)
            output_info['epoch_time'] = time.time() - epoch_start_time

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)

            # add checkpoint to summary file
            with open(args.summary, 'a+') as csvfile:
                read_data = csv.reader(csvfile)
                writer = csv.writer(csvfile)

                # write header row
                if os.path.getsize(args.summary) == 0:
                    writer.writerow(output_info.keys())

                writer.writerow(output_info.values())

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == '__main__':
    main()
