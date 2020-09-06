import torch.nn.functional as F
import torch.nn as nn
import torch
from torchsummary import summary


class Encoder(nn.Module):
    """
        function : GanNormal Encoding NetWork

        input  : org_channel * size * size
        output : finnal_channel * 1 * 1
    """

    def __init__(
            self,
            size,
            finnal_channel,
            org_channel,
            conv_channel,
            ngpu,
            n_extra_layers=0,
            add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        # 图片大小应当为16的倍数
        assert size % 16 == 0
        # 421配置图片缩小一倍
        model = nn.Sequential()
        model.add_module(
            'inittal-conv-{0}-{1}'.format(
                org_channel, conv_channel), nn.Conv3d(
                org_channel, conv_channel, [
                    3, 4, 4], 2, 1, bias=False))
        model.add_module('initial-relu-{0}'.format(conv_channel),
                         nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = size / 2, conv_channel

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            model.add_module(
                'pyramid-{0}-{1}-conv'.format(
                    in_feat, out_feat), nn.Conv3d(
                    in_feat, out_feat, [
                        3, 4, 4], 2, 1, bias=False))
            model.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                             nn.BatchNorm3d(out_feat))
            model.add_module('pyramid-{0}-relu'.format(out_feat),
                             nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        if add_final_conv:
            model.add_module(
                'final-{0}-{1}-conv'.format(
                    cndf, 1), nn.Conv3d(
                    cndf, finnal_channel, [
                        1, 4, 4], 1, 0, bias=False))

    #    print (model)
        self.modle = model

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.modle, input, range(self.ngpu))
        else:
            output = self.modle(input)

        return output


class Decoder (nn.Module):
    """
        function : GanNormal Decoding NetWork
    """

    def __init__(
            self,
            size,
            finnal_channel,
            org_channel,
            conv_channel,
            ngpu,
            n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert size % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = conv_channel // 2, 4
        while tisize != size:
            cngf = cngf * 2
            tisize = tisize * 2

        model = nn.Sequential()
        model.add_module(
            'initial-{0}-{1}-convt'.format(
                finnal_channel, cngf), nn.ConvTranspose3d(
                finnal_channel, cngf, [
                    1, 4, 4], 1, 0, bias=False))
        model.add_module('initial-{0}-batchnorm'.format(cngf),
                         nn.BatchNorm3d(cngf))
        model.add_module('initial-{0}-relu'.format(cngf),
                         nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < size // 2:
            model.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                             nn.ConvTranspose3d(cngf, cngf // 2, [3, 4, 4], 2, 1, bias=False))
            model.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                             nn.BatchNorm3d(cngf // 2))
            model.add_module('pyramid-{0}-relu'.format(cngf // 2),
                             nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        model.add_module(
            'final-{0}-{1}-convt'.format(
                cngf, org_channel), nn.ConvTranspose3d(
                cngf, org_channel, [
                    6, 4, 4], 2, 1, bias=False))
        model.add_module('final-{0}-tanh'.format(org_channel),
                         nn.Tanh())
        self.model = model

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()
        module = Encoder(
            args.size,
            args.output_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        layers = list(module.modle.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class NetG(nn.Module):
    def __init__(self, args):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        self.decoder = Decoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        self.encoder2 = Encoder(
            args.size,
            args.finnal_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_images = self.decoder(latent_i)
        latent_o = self.encoder2(gen_images)
        return latent_i, gen_images, latent_o


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()
        module = Encoder(
            args.size,
            args.output_channel,
            args.org_channel,
            args.conv_channel,
            args.ngpu,
            args.n_extra_layers)
        layers = list(module.modle.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


if __name__ == '__main__':
    import sys
    import os
    # __file__获取执行文件相对路径，整行为取上一级的上一级目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    print(BASE_DIR)
    from parse.parse import parse_args

    args = parse_args()

    data = torch.rand(16, 3, 4, 64, 64).cuda()
    print(data.shape)
    encoder = Encoder(
        args.size,
        args.finnal_channel,
        args.org_channel,
        args.conv_channel,
        args.ngpu,
        args.n_extra_layers).cuda()
    summary(encoder, (3, 4, 64, 64))

    latent_i = encoder.forward(data)
    print(latent_i.shape)

    decoder = Decoder(
        args.size,
        args.finnal_channel,
        args.org_channel,
        args.conv_channel,
        args.ngpu,
        args.n_extra_layers).cuda()

    summary(decoder, (100, 1, 1, 1))

    result = decoder.forward(latent_i)
    print(result.shape)
