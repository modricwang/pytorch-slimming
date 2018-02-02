import os
import cv2
import numpy
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data


def get_train_loader(args):
    dataset = VehicleTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


def get_test_loader(args):
    dataset = VehicleTestSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


class VehicleTrainSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.architecture = args.model

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

        train_list = open(os.path.join(args.data_dir, 'train_list.txt'), 'r').readlines()

        for line in train_list:
            if args.small_set:
                if random.random() < 0.9:
                    continue

            sample_info = line.split(' ')

            this_image = sample_info[0][1:-1]
            this_color = int(sample_info[1])
            this_type = int(sample_info[2][:-1])

            self.images.append(this_image)
            self.targets.append((this_color, this_type))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.architecture == 'inception':
            image = cv2.resize(image, (299, 299))
        else:
            image = cv2.resize(image, (224, 224))

        image = self.transform(image)
        return (image, self.targets[index])

    def __len__(self):
        return len(self.targets)


class VehicleTestSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.architecture = args.model

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

        test_list = open(os.path.join(args.data_dir, 'test_list.txt'), 'r').readlines()

        for line in test_list:
            sample_info = line.split(' ')

            this_image = sample_info[0][1:-1]
            this_color = int(sample_info[1])
            this_type = int(sample_info[2][:-1])

            self.images.append(this_image)
            self.targets.append((this_color, this_type))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.architecture == 'inception':
            image = cv2.resize(image, (299, 299))
        else:
            image = cv2.resize(image, (224, 224))

        image = self.transform(image)
        return (image, self.targets[index])

    def __len__(self):
        return len(self.targets)
