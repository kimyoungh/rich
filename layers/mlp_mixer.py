"""
    Fundamental NN Layers - MLP Mixer

    @author: Younghyun Kim
    Created on 2022.09.03
"""
import torch.nn as nn

from layers.mapping import Mapping


class MLPMixer(nn.Module):
    """
        MLP Mixer
    """
    def __init__(self, input_dim, channel_dim,
                seq_len, nlayers=2,
                slope=0.2, dropout=0.1):
        """
            Initialization

            Input Shape: (batch_size X seq_len X input_dim)
        """
        super().__init__()

        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.seq_len = seq_len
        self.nlayers = nlayers
        self.slope = slope
        self.dropout = dropout

        self.in_net = nn.Sequential(
            nn.Linear(input_dim, channel_dim),
            nn.LeakyReLU(slope),
        )

        self.mixer_layers = nn.ModuleList()

        for i in range(nlayers):
            layer = MLPMixerLayer(channel_dim, seq_len,
                                slope, dropout)
            self.mixer_layers.append(layer)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_in):
        """
            x_in: (batch_size X seq_len X input_dim)
            x_out: (batch_size X seq_len X channel_dim)
        """
        x_out = self.in_net(x_in)

        for i in range(self.nlayers):
            x_out = self.mixer_layers[i](x_out)

        return x_out


class MLPMixerLayer(nn.Module):
    """
        MLP Mixer Block
    """
    def __init__(self, channel_dim=16, seq_len=250,
                slope=0.2, dropout=0.1):
        """
            Initialization

            Input Shape: (batch_size X seq_len X channel_dim)
        """
        super().__init__()
        self.channel_dim = channel_dim
        self.seq_len = seq_len
        self.slope = slope
        self.dropout_p = dropout

        self.layer_norm_1 = nn.LayerNorm(channel_dim)
        self.mlp_1 = Mapping(seq_len, seq_len, 2, 'first',
                            slope, dropout, False)

        self.layer_norm_2 = nn.LayerNorm(channel_dim)
        self.mlp_2 = Mapping(channel_dim, channel_dim,
                            2, 'first', slope, dropout, False)

        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_in):
        """
            x_in: (batch_size X seq_len X channel_dim)
        """
        x_1 = self.layer_norm_1(x_in).transpose(1, 2).contiguous()
        x_1 = self.mlp_1(x_1).transpose(1, 2).contiguous()

        x_2 = x_1 + self.dropout(x_in)
        x_3 = self.mlp_2(self.layer_norm_2(x_2))

        x_out = x_3 + self.dropout(x_2)

        return x_out