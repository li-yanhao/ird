#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import math
import torch.nn as nn
import torch
import numpy as np
import os


def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))


def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()


def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)


def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    r"""
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features)==len(kernels))

    layers = list()
    for i in range(0,depth):
        if i==0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


def get_model():
    num_levels = 17
    out_channel = 3
    model = make_net(3, kernels=[3, ] * num_levels,
                        features=[64, ] * (num_levels - 1) + [out_channel],
                        bns=[False, ] + [True, ] *
                        (num_levels - 2) + [False, ],
                        acts=['relu', ] * (num_levels - 1) + ['linear', ],
                        dilats=[1, ] * num_levels,
                        bn_momentum=0.1, padding=0)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_folder, "DenoiserWeight/model_best.th")
    state_dict = torch.load(weights_path, torch.device('cpu'))
    model.load_state_dict(state_dict["network"])
    model.eval()

    return model


class DnCNN(nn.Module):
    def __init__(self, freeze:bool=True) -> None:
        super().__init__()
        self.model = get_model()
        self.freeze = freeze
    
    def forward(self, imgs):
        """
        Parameters
        ----------
        imgs : BxCxHxW

        Returns
        -------
        residuals : BxCxHxW
        """
        if self.freeze:
            with torch.no_grad():
                img_residuals: torch.Tensor = self.model(imgs)
        else:
           img_residuals: torch.Tensor = self.model(imgs)

        return img_residuals

    def extract_noise(self, img:np.ndarray):
        # assume img in [0, 255]
        img_3_channel = torch.from_numpy(img / 255)[None, None, ...].float().repeat(1, 3, 1, 1)
        residual_3_channel = self.forward(img_3_channel)
        residual = residual_3_channel[0,0,:].cpu().numpy() * 255
        return residual


def test_DnCNN():
    import numpy as np
    import iio
    img = np.random.uniform(0, 255, (600, 600))  # should be in [0, 1]
    iio.write("in_dncnn.tiff", img)

    denoiser = DnCNN()
    residual = denoiser.extract_noise(img)
    # img_3_channel = torch.from_numpy(img)[None, None, ...].float().repeat(1, 3, 1, 1)

    # residual_3_channel = denoiser.forward(img_3_channel)
    # residual = residual_3_channel[0,0,:].cpu().numpy()
    iio.write("out_dncnn.tiff", residual)


if __name__ == "__main__":
    test_DnCNN()
