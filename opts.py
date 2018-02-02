import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# General options
parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-small_set', action='store_true', help='Whether uses a small dataset')
parser.add_argument('-train_record', action='store_true', help='Path to save train record')
parser.add_argument('-test_only', action='store_true', help='Only conduct test on the validation set')

parser.add_argument('-model', required=True, help='Model type when we create a new one')
parser.add_argument('-data_dir', required=True, help='Path to data directory')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-color_classes', required=True, type=int, help='Num of color classes')
parser.add_argument('-type_classes', required=True, type=int, help='Num of type classes')

# Training options
parser.add_argument('-learn_rate', default=1e-2, type=float, help='Base learning rate of training')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=5e-4, type=float, help='Weight decay for training')
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=64, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-criterion', default='CrossEntropy', help='Type of objective function')

# Model options
parser.add_argument('-pretrained', default=None, help='Path to the pretrained model')
parser.add_argument('-resume', action='store_true', help='Whether continue to train from a previous checkpoint')
parser.add_argument('-nGPU', default=4, type=int, help='Number of GPUs for training')
parser.add_argument('-workers', default=4, type=int, help='Number of subprocesses to to load data')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
# parser.add_argument('--dataset', type=str, default='cifar10',
#                     help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
