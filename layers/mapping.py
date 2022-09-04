"""
    Fundamental NN Layers - Mapping

    @author: Younghyun Kim
    Created on 2022.09.03
"""
import torch
import torch.nn as nn


class Mapping(nn.Module):
    """
        Mapping Network
    """
    def __init__(self, in_dim, out_dim,
                num_layers=8, out_dim_pos='last',
                slope=0.2, dropout=0.1,
                last_activation=True):
        """
            Args:
                in_dim: input dim
                out_dim: output dim
                num_layers: # of layers
                out_dim_pos: out_dim을 적용할 layer 위치(default: last)
                    * first: 첫번째 layer out_dim
                    * last: 마지막 layer out_dim
                slope: negative slope for leaky relu
                dropout: dropout
                last_activation: Bool.(default: True)
                    * True: 마지막 layer에 leaky relu 적용
                    * False: 마지막 layer에 leaky relu 미적용
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.out_dim_pos = out_dim_pos
        self.slope = slope
        self.dropout = dropout
        self.last_activation = last_activation

        self.map_blocks = nn.ModuleList()

        in_d = in_dim

        for i in range(num_layers):
            if i < num_layers - 1:
                if out_dim_pos == 'last':
                    out_d = in_dim
                elif out_dim_pos == 'first':
                    out_d = out_dim
                block = MappingBlock(in_d, out_d, slope, dropout)
            else:
                out_d = out_dim
                if last_activation:
                    block = MappingBlock(in_d, out_d, slope, dropout)
                else:
                    block = nn.Linear(in_d, out_d)
                    nn.init.kaiming_normal_(block.weight)
                    if block.bias is not None:
                        with torch.no_grad():
                            block.bias.zero_()
            in_d = out_d
            self.map_blocks.append(block)

    def forward(self, x_in):
        " x_in forward "
        for i in range(self.num_layers):
            x_in = self.map_blocks[i](x_in)

        return x_in


class MappingBlock(nn.Module):
    " Default Linear Mapping Block "
    def __init__(self, in_dim, out_dim,
                leak_slope=0.2, dropout=0.1, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.leak_slope = leak_slope
        self.dropout = dropout
        self.bias = bias

        self.fc_net = nn.Linear(in_dim, out_dim, bias=bias)

        self.leaky_relu = nn.LeakyReLU(leak_slope)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.fc_net.weight)

        if self.fc_net.bias is not None:
            with torch.no_grad():
                self.fc_net.bias.zero_()

    def forward(self, x_in):
        " forward "
        x_out = self.leaky_relu(self.fc_net(x_in))
        x_out = self.dropout(x_out)

        return x_out