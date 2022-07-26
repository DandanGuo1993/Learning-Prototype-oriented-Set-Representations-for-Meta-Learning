import argparse
import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import MNIST_SeqOnline
from models import MNIST_AdderCNN
from routines import train, test, OT_loss


def main():
    parser = argparse.ArgumentParser(description='Sum MNIST digits.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', default=False, metavar='', help='train the model')
    parser.add_argument('-m', '--model', default='./mnist_adder_cnn_ep_10.pth', metavar='', help='model file (output of train, input of test)')
    parser.add_argument('-d', '--data_path', default='./infimnist_data', metavar='', help='data directory path')
    parser.add_argument('--n_train', default=None, metavar='', help='number of training examples')
    parser.add_argument('--n_test', default=5000, metavar='', help='number of test examples')
    parser.add_argument('--n_valid', default=1000, metavar='', help='number of validation examples')
    parser.add_argument('--max_size_train', default=10, metavar='', help='maximum size of training sets')
    parser.add_argument('--min_size_test', default=20, metavar='', help='minimum size of test sets')
    parser.add_argument('--max_size_test', default=100, metavar='', help='maximum size of test sets')
    parser.add_argument('--lr', default=1e-3, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=100, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=128, metavar='', help='batch size')
    parser.add_argument('--use_cuda', default=True, metavar='', help='use CUDA capable GPU')
    args = vars(parser.parse_args())

    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'

    dataset = MNIST_SeqOnline  
    model = MNIST_AdderCNN()  
    model.to(device)
    loss = F.l1_loss
    optimizer = Adam(model.parameters(), lr=args['lr'])
    print(model)

    if args['train']:
        if args['n_train']:
            train_set = dataset(pack=0,
                                num_examples=args['n_train'],
                                max_seq_len=args['max_size_train'])
        else:
            train_set = dataset(pack=0,
                                max_seq_len=args['max_size_train'])
        train_loader = DataLoader(train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=True)

        if args['n_valid'] > 0:
            valid_set = dataset(pack=1,
                                num_examples=args['n_valid'],
                                max_seq_len=args['max_size_train'])
            valid_loader = DataLoader(valid_set,
                                      batch_size=args['batch_size'],
                                      shuffle=False)
        else:
            valid_loader = None


        print('Train on {} samples, validate on {} samples'.format(
            len(train_set), len(valid_set)))

        train(model, loss, optimizer, args['epochs'], train_loader,
              valid_loader=valid_loader, device=device, visdom=None, model_path=args['model'])

    if args['n_test'] > 0:
        model.load_state_dict(torch.load(args['model']))

        test_set = dataset(pack=np.random.randint(2, 8),
                           num_examples=args['n_test'],
                           rand_seq_len=False)
        test_loader = DataLoader(test_set,
                                 batch_size=args['batch_size'],
                                 shuffle=False)

        test(model, loss, test_loader,
             size_range=[args['min_size_test'], args['max_size_test']+1],
             device=device, visdom=None)


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
