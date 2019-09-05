################
# Taken from https://github.com/thunil/Deep-Flow-Prediction
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=[1,0], dropout=0., upsample_size=1):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=[size,1], stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(size=[upsample_size,1]))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=[(size-1),1], stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    
# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(5, channels, kernel_size=[4,1], stride=2, padding=[1,0], bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_size=2)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_size=4)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, upsample_size= 8)
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, upsample_size = 16)
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, upsample_size = 32)
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout, upsample_size = 64)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 2, [4,1], 2, padding=[1,0], bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        #print('x.shape', x.shape)
        #print('out1.shape',out1.shape)
        #print('out2.shape',out2.shape)
        #print('out2b.shape',out2b.shape)
        #print('out3.shape',out3.shape)
        #print('out4.shape',out4.shape)
        #print('out5.shape',out5.shape)
        #print('out6.shape',out6.shape)

        dout6 = self.dlayer6(out6)
        #print('dout6.shape',dout6.shape)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        #print('dout5.shape',dout5.shape)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        #print('dout4.shape',dout4.shape)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        #print('dout3.shape',dout3.shape)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        #print('dout2b.shape',dout2b.shape)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)

        #print('dout2.shape',dout2.shape)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        #print('dout1.shape',dout1.shape)
        return dout1

# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2,ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch  , ch*2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch*8, 1   , 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch*2)
        self.bnc2 = nn.BatchNorm2d(ch*4)
        self.bnc3 = nn.BatchNorm2d(ch*8)        

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h) 
        return h
