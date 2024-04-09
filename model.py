import torch.nn as nn
import torch


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


class Resnet101IbnA(nn.Module):
    def __init__(self, num_classes=576, embedding_dim=2048):
        super().__init__()
        self.resnet101_ibn_a = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
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
    

if __name__ == '__main__':
    net = Resnet101IbnA()
    print(net)
