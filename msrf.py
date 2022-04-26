import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

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
        self.ny1t = nn.Sequential(nn.ConvTranspose2d(in_ch_y, gc, kernel_size = (4,4), stride = (2, 2), padding = (1,1), bias = True),
        nn.LeakyReLU(negative_slope = 0.25))

        self.nx2 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1),bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny2 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc, gc, kernel_size = (3,3), stride = (1,1), padding = (1,1),bias = bias),
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

        self.nx5 = nn.Sequential(nn.Conv2d(in_ch_x + gc + gc + gc + gc + gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
        nn.LeakyReLU(negative_slope = 0.25))

        self.ny5 = nn.Sequential(nn.Conv2d(in_ch_y + gc + gc + gc + gc + gc, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = bias),
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

# class MSRF(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):

"""
Function to test Encoder 1
"""
def test_encoder1():
    x = torch.ones(3,3,256,256)
    encoder = Encoder(3,init_feat = 32)
    #testing shape of final encoder output
    print(encoder(x)[3].shape)



if __name__ == "__main__":
    test_encoder1()