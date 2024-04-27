import torch.nn as nn
import torch

import os
import torch.nn.functional as F
from collections import OrderedDict
import DenseNet


__all__ = ['IBN_A', 'resnet101_ibn_a', 'resnext101_ibn_a', 'densenet169_ibn_a', 'se_resnet101_ibn_a']

model_urls = {
    'densenet169_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v1-hubconf/IBN_densenet.pth',
    'se_resnet101_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v1-hubconf/IBN_seresnet.pth',
    'resnext101_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v1-hubconf/IBN_resnext.pth',
    # 'resnet101_ibn_a': ,
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# this class is kept for compatibility issue run3, run4 and rerun use this class 
class Resnet101IbnA(nn.Module):
    def __init__(self, num_classes=576):
        from warnings import warn
        warn('Deprecated warning: You should only use this class if you want to load the model trained in older commits. You should use `make_model(backbone, num_classes)` to build the model in newer version.')

        super().__init__()
        self.resnet101_ibn_a = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        
        embedding_dim = self.resnet101_ibn_a.fc.in_features
        
        self.resnet101_ibn_a.fc = nn.Identity() # pretend the last layer does not exist



        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.resnet101_ibn_a(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out



class IBN_A(nn.Module):
    def __init__(self, backbone, pretrained=True, num_classes=576, embedding_dim=2048):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained=pretrained)

        # the expected embedding space is \mathbb{R}^{2048}. resnet, seresnet, resnext satisfy this automatically
        if backbone == 'densenet':
            self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, embedding_dim)
        else:
            self.backbone.fc = nn.Identity() # pretend the last layer does not exist


        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.backbone(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    



def get_backbone(backbone, pretrained):
    print(f'using {backbone} as backbone')
    
    assert backbone in ['resnet', 'resnext', 'seresnet', 'densenet'], "no such backbone, we only support ['resnet', 'resnext', 'seresnet', 'densenet']"

    if backbone == 'resnet':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=pretrained)
    
    if backbone == 'resnext':
        return torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=pretrained)

    if backbone == 'seresnet':
        return torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=pretrained)


    if backbone == 'densenet':
        return DenseNet.densenet169_ibn_a(pretrained=pretrained)
    

def make_model(backbone, num_classes):
    return IBN_A(backbone, num_classes)




def densenet169_ibn_a(print_net=False):
    model = IBN_A(backbone='densenet', pretrained=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet169_ibn_a']))

    if print_net:
        print(model)

    return model



def se_resnet101_ibn_a(print_net=False):
    model = IBN_A(backbone='seresnet', pretrained=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['se_resnet101_ibn_a']))

    if print_net:
        print(model)

    return model


def resnext101_ibn_a(print_net=False):
    model = IBN_A(backbone='resnext', pretrained=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnext101_ibn_a']))

    if print_net:
        print(model)

    return model


def resnet101_ibn_a(print_net=False):
    model = IBN_A(backbone='resnet', pretrained=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['']))

    if print_net:
        print(model)

    return model