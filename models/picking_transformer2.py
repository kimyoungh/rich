"""
    Module for Asset Picking Transformer with Cash

    @author: Younghyun Kim
    Created on 2023.02.11
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mapping import Mapping
from layers.transformer import TransformerEnc
from models.cfg.picking_transformer2_config\
    import PICKING_TRANSFORMER2_CONFIG


class PickingTransformer2(nn.Module):
    """
        Picking Transformer
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = PICKING_TRANSFORMER2_CONFIG
        self.config = config

        self.seq_len = config['seq_len']
        self.d_model = config['d_model']
        self.dim_ff = config['dim_ff']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.close_embeds_nlayers = config['close_embeds_nlayers']
        self.value_time_embeds_nlayers =\
            config['value_time_embeds_nlayers']
        self.value_cs_embeds_nlayers = config['value_cs_embeds_nlayers']
        self.fusion_embeds_nlayers =\
            config['fusion_embeds_nlayers']
        self.allocator_map_nlayers = config['allocator_map_nlayers']

        # Cash Embeddings
        self.cash_embeds = nn.Embedding(1, self.d_model)

        # Close Embeddings
        self.close_embeds = Mapping(self.seq_len, self.d_model,
                                    self.close_embeds_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Value Time Embeddings
        self.value_time_embeds = Mapping(self.seq_len, self.d_model,
                                    self.value_time_embeds_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Value CS Embeddings
        self.value_cs_embeds = Mapping(self.seq_len, self.d_model,
                                    self.value_cs_embeds_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Decision Transformer
        self.dt = TransformerEnc(self.d_model, self.nhead, self.nlayers,
                                self.dim_ff, self.dropout,
                                self.activation, True)

        # Fusion Embeddings
        self.fusion_embeds = Mapping(self.d_model * 3, self.d_model,
                                    self.fusion_embeds_nlayers, 'first',
                                    self.slope, self.dropout, True)

        # Allocator
        self.allocator = Mapping(self.d_model, 1,
                                self.allocator_map_nlayers, 'last',
                                self.slope, self.dropout, False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, obs_in, softmax=True,
                seq_mask=True, src_key_padding_mask=None):
        """
            forward

            Args:
                obs_in: observations for Asset Factors
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, factor_num, seq_len)
                    * factor index
                        * 0: close
                        * 1: value time
                        * 2: value cs
                softmax: whether do softmax or not
                    * default: True
            Returns:
                preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num+1)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num+1, d_model)
        """
        batch_size, asset_num, factor_num, seq_len = obs_in.shape

        # Close Embeddings
        close_embeds = self.close_embeds(obs_in[:, :, 0])

        # Value Time Embeddings
        value_time_embeds = self.value_time_embeds(obs_in[:, :, 1])

        # Value CS Embeddings
        value_cs_embeds = self.value_cs_embeds(obs_in[:, :, 2])

        # Fusion Embeddings
        fusion_embeds = self.fusion_embeds(
            torch.cat(
                (close_embeds, value_time_embeds, value_cs_embeds),
                dim=-1))

        # Cash Embedding
        cash_embeds = self.cash_embeds(torch.tensor(
            [0]).to(self.device)).unsqueeze(0)
        cash_embeds = cash_embeds.repeat(batch_size, 1, 1)

        embeds = torch.cat((cash_embeds, fusion_embeds), dim=1)

        outputs = self.dt(embeds, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        preds = self.allocator(outputs).view(batch_size, asset_num+1)

        if softmax:
            return preds.softmax(dim=-1), outputs
        else:
            return preds, outputs