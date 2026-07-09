import torch
import torch.nn as nn
from difflogic import LogicConvLayer, BinaryEncodingLayer

class LogicBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super(LogicBlock, self).__init__()
        llkw = dict(grad_factor=opt.grad_factor, connections=opt.connections, implementation=opt.implementation)
        
        self.conv_1 = LogicConvLayer(in_dim=in_channels, out_dim=out_channels, tree_d=3, rf=1, padding=False, **llkw)
        self.conv_2 = LogicConvLayer(in_dim=out_channels, out_dim=out_channels, tree_d=3, rf=3, padding=True, **llkw)
        self.conv_3 = LogicConvLayer(in_dim=out_channels, out_dim=out_channels, tree_d=3, rf=1, padding=False, **llkw)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        return out

class LogicNet(nn.Module):
    def __init__(self, opt, channels):
        super(LogicNet, self).__init__()
        self.conv_features = opt.channels
        llkw = dict(grad_factor=opt.grad_factor, connections=opt.connections, implementation=opt.implementation)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

        self.inc = LogicConvLayer(in_dim=channels, out_dim=self.conv_features, tree_d=3, rf=3, padding=True, **llkw)
        self.down1 = LogicBlock(opt, 4*self.conv_features, 2*self.conv_features)
        self.down2 = LogicBlock(opt, 8*self.conv_features, 4*self.conv_features)
        self.down3 = LogicBlock(opt, 16*self.conv_features, 8*self.conv_features)
        self.up1 = LogicBlock(opt, 6*self.conv_features, 4*self.conv_features)
        self.up2 = LogicBlock(opt, 3*self.conv_features, 2*self.conv_features)
        self.up3 = LogicBlock(opt, int(1.5*self.conv_features), self.conv_features)
        
        self.out_last = LogicConvLayer(in_dim=self.conv_features, out_dim=self.conv_features, tree_d=3, rf=3, padding=True, **llkw)

        self.b_encoder = BinaryEncodingLayer()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_enc = self.b_encoder(x)

        # down
        x1 = self.inc(x_enc)
        x2 = self.down1(self.pixel_unshuffle(x1))
        x3 = self.down2(self.pixel_unshuffle(x2))
        x4 = self.down3(self.pixel_unshuffle(x3))
        # up
        x_up = self.up1(torch.cat([self.pixel_shuffle(x4), x3], dim=1))
        x_up = self.up2(torch.cat([self.pixel_shuffle(x_up), x2], dim=1))
        x_up = self.up3(torch.cat([self.pixel_shuffle(x_up), x1], dim=1))
        # out
        logits = (self.out_last(x_up).sum(dim=1, keepdim=True)-0.5*self.conv_features) / (0.5*self.conv_features)
        return logits

class LogicNetTiny(nn.Module):
    def __init__(self, opt, channels):
        super(LogicNetTiny, self).__init__()
        self.conv_features = opt.channels
        self.block_features = max(1, self.conv_features // 8)
        llkw = dict(grad_factor=opt.grad_factor, connections=opt.connections, implementation=opt.implementation)

        self.inc = LogicConvLayer(in_dim=channels, out_dim=self.block_features, tree_d=3, rf=3, padding=True, **llkw)
        self.block1 = LogicBlock(opt, self.block_features, self.block_features)
        self.block2 = LogicBlock(opt, self.block_features, self.block_features)
        self.block3 = LogicBlock(opt, self.block_features, self.block_features)
        self.out_last = LogicConvLayer(in_dim=self.block_features, out_dim=self.conv_features, tree_d=3, rf=3, padding=True, **llkw)

        self.b_encoder = BinaryEncodingLayer()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.b_encoder(x)
        x = self.inc(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = (self.out_last(x).sum(dim=1, keepdim=True)-0.5*self.conv_features) / (0.5*self.conv_features)
        return logits
