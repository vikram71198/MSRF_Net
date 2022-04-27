import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision.models.resnet import BasicBlock

"""
Squeeze & Excitation Block
"""
class SE_Block(nn.Module):
    def __init__(self, in_ch, ratio = 16):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(in_ch, in_ch//ratio), nn.ReLU(), nn.Linear(in_ch//ratio, in_ch), nn.Sigmoid())
    
    def forward(self, x):
        y = x.mean((-2,-1))
        y = self.block(y).unsqueeze(-1).unsqueeze(-1)
        return x*y
"""
Encoder Block
"""
class Encoder(nn.Module):
    def __init__(self, in_ch, init_feat = 32):

        super().__init__()

        '''Instantiations of all subclasses of nn.Module will be callable objects because nn.Module has __call__() built-in (which is inherited by the subclass)
        which in turn calls forward(). So, if forward() is overridden in the subclass, the new forward() will be called!'''

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, init_feat, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(), nn.Conv2d(init_feat, init_feat, kernel_size = (3,3), stride = (1,1), padding = (1,1)), nn.ReLU(),
        nn.BatchNorm2d(init_feat), SE_Block(init_feat, ratio = init_feat // 2))

        self.enc2 = nn.Sequential(nn.MaxPool2d(kernel_size = (2,2)), nn.Dropout(0.2),
        nn.Conv2d(init_feat, init_feat*2, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.Conv2d(init_feat*2, init_feat*2, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.BatchNorm2d(init_feat*2), SE_Block(init_feat*2, ratio = init_feat // 2))

        self.enc3 = nn.Sequential(nn.MaxPool2d(kernel_size = (2,2)), nn.Dropout(0.2),
        nn.Conv2d(init_feat*2, init_feat*4, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.Conv2d(init_feat*4, init_feat*4, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.BatchNorm2d(init_feat*4), SE_Block(init_feat*4, ratio = init_feat // 2))

        self.enc4 = nn.Sequential(nn.MaxPool2d(kernel_size = (2,2)), nn.Dropout(0.2),
        nn.Conv2d(init_feat*4, init_feat*8, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.Conv2d(init_feat*8, init_feat*8, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
        nn.ReLU(),
        nn.BatchNorm2d(init_feat*8))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x1, x2, x3, x4
        
"""
DSDF (Dual Scale Dense Fusion) Block
"""
class DSDF(nn.Module):
    def __init__(self, in_ch_x, in_ch_y, nf1 = 128, nf2 = 256, gc = 64, bias = True):
        super().__init__()

        self.nx1 = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1),bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny1 = nn.Sequential(nn.Conv2d(in_ch_y, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.nx1c = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.ny1t = nn.Sequential(nn.ConvTranspose2d(in_ch_y, gc, kernel_size = (4,4), stride = (2, 2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.nx2 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1),bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny2 = nn.Sequential(nn.Conv2d(in_ch_y + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1),bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.nx2c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.ny2t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.nx3 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny3 = nn.Sequential(nn.Conv2d(in_ch_y + gc + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.nx3c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.ny3t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.nx4 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny4 = nn.Sequential(nn.Conv2d(in_ch_y + gc + gc + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.nx4c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        #TO DO
        self.ny4t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size = (4,4), stride = (2,2), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.nx5 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc + gc + gc + gc, nf1, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny5 = nn.Sequential(nn.Conv2d(in_ch_y + gc + gc + gc + gc + gc, nf2, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))


    def forward(self, x, y):
        x1 = self.nx1(x)
        y1 = self.ny1(y)

        x1c = self.nx1c(x)
        y1t = self.ny1t(y)

        x2_input = torch.cat([x, x1, y1t], dim = 1)
        x2 = self.nx2(x2_input)

        y2_input = torch.cat([y, y1, x1c], dim = 1)
        y2 = self.ny2(y2_input)

        x2c = self.nx2c(x1)
        y2t = self.ny2t(y1)

        x3_input = torch.cat([x, x1, x2, y2t], dim = 1)
        x3 = self.nx3(x3_input)

        y3_input = torch.cat([y, y1, y2, x2c], dim = 1)
        y3 = self.ny3(y3_input)

        x3c = self.nx3c(x2)
        y3t = self.ny3t(y2)

        x4_input = torch.cat([x, x1, x2, x3, y3t], dim = 1)
        x4 = self.nx4(x4_input)

        y4_input = torch.cat([y, y1, y2, y3, x3c], dim = 1)
        y4 = self.ny4(y4_input)

        x4c = self.nx4c(x3)
        y4t = self.ny4t(y3)

        x5_input = torch.cat([x, x1, x2, x3, x4, y4t], dim = 1)
        x5 = self.nx5(x5_input)

        y5_input = torch.cat([y, y1, y2, y3, y4, x4c], dim = 1)
        y5 = self.ny5(y5_input)
        
        x5 *= 0.4
        y5 *= 0.4

        return x5 + x, y5 + y
"""
MSRF Sub-Network implementing Multi-Scale Fusion using DSDF Blocks
"""
class MSRF_SubNet(nn.Module):
    def __init__(self, init_feat):
        super().__init__() 
        self.dsfs_1  = DSDF(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_2  = DSDF(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_3  = DSDF(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_4  = DSDF(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_5  = DSDF(init_feat*2, init_feat*4, nf1=init_feat*2, nf2=init_feat*4, gc=init_feat*2//2)
        self.dsfs_6  = DSDF(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_7  = DSDF(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_8  = DSDF(init_feat*2, init_feat*4, nf1=init_feat*2, nf2=init_feat*4, gc=init_feat*2//2)
        self.dsfs_9  = DSDF(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_10 = DSDF(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)

    def forward(self, x11, x21, x31, x41):
        x12, x22 = self.dsfs_1(x11, x21)
        x32, x42 = self.dsfs_2(x31, x41)
        x12, x22 = self.dsfs_3(x12, x22)
        x32, x42 = self.dsfs_4(x32, x42)
        x22, x32 = self.dsfs_5(x22, x32)
        x13, x23 = self.dsfs_6(x12, x22)
        x33, x43 = self.dsfs_7(x32, x42)
        x23, x33 = self.dsfs_8(x23, x33)
        x13, x23 = self.dsfs_9(x13, x23)
        x33, x43 = self.dsfs_10(x33, x43)

        x13 = (x13*0.4) + x11
        x23 = (x23*0.4) + x21
        x33 = (x33*0.4) + x31
        x43 = (x43*0.4) + x41

        return x13, x23, x33, x43

"""
Gated Convolutions
"""
class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out
"""
Shape Stream
"""
class ShapeStream(nn.Module):
    def __init__(self, init_feat):
        super().__init__()
        self.res2_conv = nn.Conv2d(init_feat * 2, 1, 1)
        self.res3_conv = nn.Conv2d(init_feat * 4, 1, 1)
        self.res4_conv = nn.Conv2d(init_feat * 8, 1, 1)
        self.res1 = BasicBlock(init_feat, init_feat, 1)
        self.res2 = BasicBlock(32, 32, 1)
        self.res3 = BasicBlock(16, 16, 1)
        self.res1_pre = nn.Conv2d(init_feat, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32)
        self.gate2 = GatedConv(16, 16)
        self.gate3 = GatedConv(8, 8)
        self.gate = nn.Conv2d(8, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)
    
    def forward(self, x, res2, res3, res4, grad):
        size = grad.shape[-2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        res2 = F.interpolate(self.res2_conv(res2), size, mode='bilinear', align_corners=True)
        res3 = F.interpolate(self.res3_conv(res3), size, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(res4), size, mode='bilinear', align_corners=True)
        gate1 = self.gate1(self.res1_pre(self.res1(x)), res2)
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res3)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res4)
        gate = torch.sigmoid(self.gate(gate3))
        feat = torch.sigmoid(self.fuse(torch.cat((gate, grad), dim=1)))
        return gate, feat

class MSRF(nn.Module):
    def __init__(self, in_ch, n_classes, init_feat = 32):
        super().__init__()
        self.encoder1 = Encoder(in_ch, init_feat)
        self.msrf_subnet = MSRF_SubNet(init_feat)
        self.shape_stream = ShapeStream(init_feat)

    def forward(self, x, canny):
        e1, e2, e3, e4 = self.encoder1(x)
        x13, x23, x33, x43 = self.msrf_subnet(e1, e2, e3, e4)
        canny_gate, canny_feat = self.shape_stream(x13, x23, x33, x43, canny)
        return canny_gate, canny_feat

"""
Function to test Encoder 1
"""
def test_encoder1():
    x = torch.ones(3,3,256,256)
    encoder = Encoder(3,init_feat = 32)
    #testing shape of final encoder output
    print(encoder(x)[3].shape)

"""
Function to test MSRF-NET 
"""
def test_msrf():
    with torch.no_grad():
        x = torch.randn((2,1,256,256))
        canny = torch.randn((2,1,256,256))
        msrf = MSRF(1, 3, init_feat = 32)
        print(msrf(x, canny)[0].shape)

if __name__ == "__main__":
    test_msrf()