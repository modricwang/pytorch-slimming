from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from opts import args
from datasets import get_train_loader
from datasets import get_test_loader
from log import Logger
from vgg import vgg
import shutil
from train import Trainer
import torch.backends.cudnn as cudnn

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('/mnt/lustre/86share3/zhongzhao/data/dqdata/cifar10', train=True,
#                      transform=transforms.Compose([
#                          transforms.RandomCrop(32, padding=4),
#                          transforms.RandomHorizontalFlip(),
#                          transforms.ToTensor(),
#                          transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
#                      ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


train_loader = get_train_loader(args)
test_loader = get_test_loader(args)

#
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('/mnt/lustre/86share3/zhongzhao/data/dqdata/cifar10', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
#     ])),
#     batch_size=args.test_batch_size, shuffle=False, **kwargs)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = vgg(cfg=checkpoint['cfg'], args=args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = vgg(args=args)
    model = torch.nn.DataParallel(model)
    if args.cuda:
        model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def create_model(args):
    state = None

    model_creators = vgg

    assert args.model in model_creators

    model = model_creators[args.model](args)

    if args.resume:
        save_path = os.path.join(args.save_path, args.model)

        if args.small_set:
            save_path += '-Small'
        else:
            save_path += '-Baseline'

        print("=> Loading checkpoint from " + save_path)
        assert os.path.exists(save_path), "[!] Checkpoint " + save_path + " doesn't exist"

        latest = torch.load(os.path.join(save_path, 'latest.pth'))
        latest = latest['latest']

        checkpoint = os.path.join(save_path, 'model_%d.pth' % latest)
        checkpoint = torch.load(checkpoint)

        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    if args.nGPU > 0:
        cudnn.benchmark = True
        if args.nGPU > 1:
            model = nn.DataParallel(model, device_ids=[i for i in range(args.nGPU)]).cuda()
        else:
            model = model.cuda()

    criterion = nn.__dict__[args.criterion + 'Loss']()
    if args.nGPU > 0:
        criterion = criterion.cuda()

    return model, criterion, state


# Create Model, Criterion and State
model, criterion, state = create_model(args)
print("=> Model and criterion are ready")
# Create Dataloader
if not args.test_only:
    train_loader = get_train_loader(args)
val_loader = get_test_loader(args)
print("=> Dataloaders are ready")
# Create Logger
logger = Logger(args, state)
print("=> Logger is ready")  # Create Trainer
trainer = Trainer(args, model, criterion, logger)
print("=> Trainer is ready")

if args.test_only:
    test_summary = trainer.test(0, val_loader)
    print("- Test:  Color %6.3f  Type %6.3f" % (
        test_summary['color_acc'],
        test_summary['type_acc']))
else:
    start_epoch = logger.state['epoch'] + 1
    print("=> Start training")

    for epoch in range(start_epoch, args.n_epochs + 1):
        train_summary = trainer.train(epoch, train_loader)
        test_summary = trainer.test(epoch, val_loader)

        logger.record(epoch, train_summary, test_summary, model)

    logger.final_print()
