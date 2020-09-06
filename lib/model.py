from lib.module import Encoder, Decoder, NetD
from lib.loss import GANLoss
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import itertools
import torch




class GANomaly3D(nn.Module):
    def __init__(self, args):

        super().__init__()
        # Store the variable

        # Define the hyper-parameters
        self.w_adv = args.w_adv
        self.w_con = args.w_con
        self.w_enc = args.w_enc
        # Define the network structure
        self.G_E = Encoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        self.G_D = Decoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        self.E = Encoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        self.f = NetD(args)
        self.l1l_criterion = nn.L1Loss(reduction='sum')
        self.l2l_criterion = nn.MSELoss(reduction='sum')
        self.bce_criterion = GANLoss(use_lsgan=False)
        self.optim_G = Adam(
            itertools.chain(
                self.G_E.parameters(),
                self.G_D.parameters(),
                self.E.parameters()),
            lr=0.0002)
        self.optim_D = Adam(self.f.parameters(), lr=0.0001)



    def forward(self, x):

        self.x = x
        self.z = self.G_E(self.x)
        self.x_ = self.G_D(self.z)
        self.z_ = self.E(self.x_)
        return self.z, self.z_

    def backward(self):

        self.optim_D.zero_grad()
        true_pred = self.f(self.x)
        fake_pred = self.f(self.x_.detach())
        self.loss_D = self.bce_criterion(
            true_pred, True) + self.bce_criterion(fake_pred, False)
        self.loss_D.backward()
        self.optim_D.step()

        self.optim_G.zero_grad()
        fake_pred = self.f(self.x_)
        self.loss_G = self.bce_criterion(fake_pred, True) * self.w_adv + \
            self.l2l_criterion(self.x_, self.x) * self.w_con + \
            self.l1l_criterion(self.z_, self.z) * self.w_enc
        self.loss_G.backward()
        self.optim_G.step()

    def getLoss(self):
        """
            Return the loss value

            Ret:    The generator loss and discriminator loss
        """
        return round(self.loss_G.item(), 5), round(self.loss_D.item(), 5)

    def getImg(self):
        """
            Return the images

            Ret:    The input image and reconstructed image
        """
        return self.x, self.x_


if __name__ == '__main__':
    import sys
    import os
    from torch.utils.data import DataLoader
    # __file__获取执行文件相对路径，整行为取上一级的上一级目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    print(BASE_DIR)
    from parse.parse import parse_args
    from lib.dataloader import VideoDataset

    args = parse_args()

    model = GANomaly3D(args=args)
    model = model.cuda()

    train_data = VideoDataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    for i, sample in enumerate(train_loader):
        inputs = sample[0].cuda()
        labels = sample[1]
        print(inputs.size())
        print(labels)

        z, z_ = model.forward(inputs)

        print(z.shape, z_.shape)

        if i == 1:
            break
