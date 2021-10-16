import torch
from torch import nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, channels, res_num):
        super(Encoder, self).__init__()

        # Initial convolution block
        out_features = 4
        self.model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(3):
            out_features *= 2
            self.model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(res_num):
            self.model += [ResidualBlock(out_features)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, channels, res_num):
        super(Decoder, self).__init__()

        self.model = []

        out_features = 32
        in_features = 32
        # Upsampling
        for _ in range(3):
            out_features //= 2
            self.model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        self.model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


# class Coders(nn.Module):
#     def __init__(self, feature_num, res_num):
#         super(Coders, self).__init__()
#         self.feature_num = feature_num
#         self.encoders = nn.ModuleList()
#         for i in range(feature_num):
#             self.encoders.append(Encoder(res_num))
#         self.decoder = Decoder(res_num)
#         self.W = torch.nn.Parameter(torch.randn(4, 1), requires_grad=True)
#
#     def forward(self, x):
#         dev = torch.device('cuda:0')
#         code = torch.zeros(size=(x.size(0), 12, self.feature_num), requires_grad=False, device=dev)
#         for i in range(0, self.feature_num):
#             code[:, :, i] = self.encoders[i](x)
#         pred_code = torch.matmul(code, self.W)
#         pred_img = self.decoder(pred_code.view(pred_code.size()[0], pred_code.size()[1]))
#         return pred_img, code


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, features_num,  input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        self.feature_num = features_num
        self.encoders = nn.ModuleList()
        for i in range(features_num):
            self.encoders.append(Encoder(input_shape[0], num_residual_blocks))
        self.dec = Decoder(input_shape[0], num_residual_blocks)
        self.W = torch.nn.Parameter(torch.randn(4, 1), requires_grad=True)

    def forward(self, x):
        dev = torch.device('cuda:0')
        code = torch.zeros(size=(x.size(0), 32, 4, 4, self.feature_num), requires_grad=False, device=dev)
        for i in range(0, self.feature_num):
            code[:, :, i] = self.encoders[i](x)
        pred_code = torch.matmul(code, self.W)
        pred_img = self.dec(pred_code.view(pred_code.size()[0], 32, 4, 4))
        return pred_img, pred_code


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    imgs = torch.randn(size=(32, 3, 32, 32), device=dev)
    # enc = Encoder(4)
    # dec = Decoder(4)
    # code = enc(imgs)
    # pred_imgs = dec(code)
    # print(pred_imgs.size())
    coder = GeneratorResNet(4, (3, 32, 32), 3).to(dev)
    print(coder(imgs).size())
