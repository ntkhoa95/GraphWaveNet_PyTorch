import sys, torch
from turtle import forward
from numpy import c_
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()
    
class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        positions = torch.arange(input_sequences.size(2)).unsqueeze(1).to(input_sequences.device) # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(input_sequences.device) # [1, d_model]
        angles = self.get_angles(positions, indexes) # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2]) # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2]) # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1) # [batch_size, seq_len, d_model]
        position_encoding= position_encoding.unsqueeze(1).repeat(1, 2, 1, 1)

        return position_encoding

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)

        return h

class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, \
                    gcn_bool=True, addaptadj=True, aptinit=None, \
                        in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256,\
                            end_channels=512, kernel_size=2, blocks=4, layers=2, apt_size=10):
        super(GWNet, self).__init__()
        self.position_enc = PositionalEncoding(d_model=13, max_len=207)
        self.layer_norm = nn.LayerNorm(13, eps=1e-6)

        self.dropout = dropout
        self.blocks  = blocks
        self.layers  = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0

        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, apt_size).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(apt_size, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size]**0.5))
                initemb2 = torch.mm(torch.diag(p[:apt_size]**0.5), n[:, :apt_size].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,\
                                                    out_channels=dilation_channels,\
                                                    kernel_size=(1, kernel_size),
                                                    dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,\
                                                    out_channels=dilation_channels,
                                                    kernel_size=(1, kernel_size),
                                                    dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,\
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,\
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(c_in=dilation_channels, c_out=residual_channels, dropout=dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,\
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,\
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        # print(self.receptive_field)

    def forward(self, input):
        # Input shape is [batch_size, features, n_nodes, n_timesteps]
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        
        x_enc = self.position_enc(x)
        x += x_enc
        x = self.layer_norm(x)

        x = self.start_conv(x)
        skip = 0

        # Calculate the current adaptive adjacency matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK
            #            | ---------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + --> *input*
            #            |    |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + -------------> *skip*

            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate   = torch.sigmoid(self.gate_convs[i](residual))
            x      = filter * gate

            # parameterized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    graph_out = self.gconv[i](x, new_supports)
                else:
                    graph_out = self.gconv[i](x, self.supports)
                # x = x + graph_out
                x = graph_out
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x