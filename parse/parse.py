from lib.utils import INFO, showParameters
import argparse
import torch
import os

"""
    This script defines the usage of each parameters

    Author: SunnerLi
"""


def parse_args(phase='train'):

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--clip_len', default=4, type=int,
                        help='seq len of video frames')
    parser.add_argument('--random', default=42, type=int,
                        help='random of system')
    parser.add_argument('--dataset', default='Le2i',
                        type=str, help='The name of dataset')

    parser.add_argument('--checkpoint', default='Le2i',
                        type=str, help='The name of dataset')

    parser.add_argument(
        '--root_dir',
        default='/home/sky/PycharmProjects/data/Video',
        type=str,
        help='The dir for orig video dir')
    parser.add_argument(
        '--pic_dir',
        default='/home/sky/PycharmProjects/data/Pic',
        type=str,
        help='The pic of video frames')
    parser.add_argument(
        '--label_dir',
        default='/home/sky/PycharmProjects/data/label',
        type=str,
        help='split dataset')
    parser.add_argument('--save_dir', default='save', type=str,
                        help='save model of training')

    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='initial num_workers, the number of processes that'
        'generate batches in parallel')

    parser.add_argument('--size', type=int, default=64,
                        help="The size of image")
    parser.add_argument('--org_channel', type=int, default=3,
                        help="The channel of image")
    parser.add_argument('--conv_channel', type=int, default=64,
                        help='The channel of conv')
    parser.add_argument('--finnal_channel', type=int, default=100,
                        help='Size of laten output')
    parser.add_argument('--output_channel', type=int, default=1,
                        help='Size of laten output')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--n_extra_layers', type=int, default=0,
                        help='number of layers to use')
    parser.add_argument('--add_final_conv', type=bool, default=True,
                        help='convert to vector')

    parser.add_argument('--w_adv', type=float, default=1,
                        help='Adversarial loss weight')
    parser.add_argument('--w_con', type=float, default=10,
                        help='Reconstruction loss weight')
    parser.add_argument('--w_enc', type=float, default=1,
                        help='Encoder loss weight.')

    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args
