from lib.visualize import visualizeEncoderDecoder
from lib.model import GANomaly3D
from parse.parse import parse_args

from lib.dataloader import VideoDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import numpy as np
import torch
import os

torch.cuda.empty_cache()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""
    This script defines the training procedure of GANomaly3D

    Author: Lc
"""


def train(args):
    """
        This function define the training process

        Arg:    args    (napmespace) - The arguments
    """
    # Create the data loader
    train_data_loader = DataLoader(
        VideoDataset(
            args=args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    # Create the model
    model = GANomaly3D(args)
    model = model.cuda()
    model.train()

    # Train!
    for epoch in range(0, args.epochs):
        bar = tqdm(train_data_loader)
        for inputs, labels in bar:
            inputs = inputs.cuda()
         #   labels = labels.cuda()
            model.forward(inputs)
            model.backward()
            loss_G, loss_D = model.getLoss()
            bar.set_description(
                "Loss_G: " +
                str(loss_G) +
                " loss_D: " +
                str(loss_D))
            bar.refresh()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, os.path.join('save', 'Ganormaly_epoch-' + str(epoch) + '.pth.tar'))
        print("Save model at {}\n".format(
            os.path.join('save', 'Ganormaly_epoch-' + str(epoch) + '.pth.tar')))

        print(len(train_data_loader), loss_G / len(train_data_loader))


if __name__ == '__main__':
    args = parse_args(phase='train')
    train(args)
