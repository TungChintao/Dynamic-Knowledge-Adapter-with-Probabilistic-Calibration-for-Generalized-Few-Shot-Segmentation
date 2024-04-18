import torch
import torch.nn.functional as F
from torch import nn

import segmentation_models_pytorch as smp


def get_model_(args) -> nn.Module:
    if args.model_name == 'PSPNet':  # and args.encoder_name == 'efficientnet-b4':
        return PSPNet(args)
    else:
        return NotImplementedError


class PSPNet(nn.Module):
    def __init__(self, args):
        super(PSPNet, self).__init__()
        assert args.get('num_classes_tr') is not None, 'Get the data loaders first'

        self.model = smp.PSPNet(encoder_name=args.encoder_name, classes=args.num_classes_tr)
        # self.encoder  output: [bsz, 4*bottleneck_dim, h1, w1]
        # self.decoder(ppm+bottleneck)  output: ppm         [bsz, 8*bottleneck_dim, h2, w2]
        #                                       bottleneck  [bsz, 4*bottleneck_dim, h3, w3]
        # self.classifier  output: [bsz, num_classes, h, w]
        self.bottleneck_dim = self.model.decoder.conv[0].out_channels

        # aux network ======================================================================
        self.aux_e = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=args.low_rank, kernel_size=1, stride=1)
        self.aux_m = nn.Conv2d(args.low_rank, args.low_rank, kernel_size=1, stride=1)
        self.aux_d = nn.Conv2d(in_channels=args.low_rank, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        
        nn.init.normal_(self.aux_e.weight, mean=0, std=1e-3)
        nn.init.normal_(self.aux_m.weight, mean=0, std=1e-3)
        nn.init.normal_(self.aux_d.weight, mean=0, std=1e-3)

        # aux network ======================================================================

        self.classifier = nn.Conv2d(self.bottleneck_dim, args.num_classes_tr, kernel_size=1,bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x_size = x.size()
        # assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        shape = (x_size[2], x_size[3])

        x = self.extract_features(x)
        x = self.LRANet(x)

        logits = self.classify(x, shape)
        return logits

    def extract_features(self, x):
        self.model.segmentation_head = torch.nn.Identity()
        x = self.model(x)
        return x

    def LRANet(self, x):
        x_e = self.aux_e(x)
        x = x + self.aux_d(self.aux_m(x_e))
        return x

    def classify(self, features, shape):
        x = self.classifier(features)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
