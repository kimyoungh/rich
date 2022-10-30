"""
    Trader Model

    @author: Younghyun Kim
    Created: 2022.10.15
"""
import torch
import torch.nn as nn

from layers.mapping import Mapping
from layers.transformer import TransformerEnc

from models.cfg.trader_config import TRADER_CONFIG


class Trader(nn.Module):
    """
        Trader Class
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = TRADER_CONFIG

        self.config = config

        self.factor_num = config['factor_num']
        self.add_factor_num = config['add_factor_num']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.factor_embeds_map_nlayers =\
            config['factor_embeds_map_nlayers']
        self.w_allocator_nlayers = config['w_allocator_nlayers']
        self.asset_list = config['asset_list']
        self.asset_num = len(self.asset_list)

        # Positional Encoding
        # 0: PE Factor
        # 1: PE Add_Factor
        # 2: Assets(Initial Investment)
        # 3: Assets(Rebalancing - Previous weights will be multiplied)
        self.positional_encoding = nn.Embedding(4, self.d_model)

        # Asset Embedding
        self.asset_embeds = nn.Embedding(self.asset_num, self.d_model)

        # Factor Embedding
        self.factor_embeds = Mapping(self.factor_num, self.d_model,
                                    self.factor_embeds_map_nlayers,
                                    'first', self.slope,
                                    self.dropout, True)

        # Additional Factor Embedding
        self.add_factor_embeds = Mapping(self.factor_num, self.d_model,
                                    self.factor_embeds_map_nlayers,
                                    'first', self.slope,
                                    self.dropout, True)

        # Transformer Encoder for Allocation
        self.attn = TransformerEnc(self.d_model, self.nhead,
                                self.nlayers, self.d_model,
                                self.dropout, self.activation, True)

        # Weights Allocator
        self.w_allocator = Mapping(self.d_model, 1,
                                self.w_allocator_nlayers,
                                'last', self.slope, self.dropout,
                                False)

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

    def forward(self, obs_in, obs_add_in=None,
                weights_rec=None,
                enc_time_mask=False, enc_key_padding_mask=None):
        """
            Inference

            Args:
                obs_in: multifactor scores data for observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, obs_num, factor_num)
                obs_add_in: multifactor scores data for additional observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, obs_add_num, factor_num)
                    * default: None
                weights_rec: recent weights
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num)
                    * default: None
            Returns:
                weights: portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num)
                outputs: transformer outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, d_model)
        """
        batch_size = len(obs_in)

        obs_in = self.factor_embeds(obs_in).mean(dim=1, keepdims=True)

        if obs_add_in is not None:
            pe_obs_idx = torch.arange(2).to(self.device)
            obs_add_in = self.add_factor_embeds(
                obs_add_in).mean(dim=1, keepdims=True)
            idx_init = 2
        else:
            pe_obs_idx = torch.tensor([0]).to(self.device)
            obs_add_in = torch.tensor([]).to(self.device)
            idx_init = 1
        pe_obs = self.positional_encoding(
            pe_obs_idx).view(1, len(pe_obs_idx), self.d_model)
        pe_obs = pe_obs.repeat(batch_size, 1, 1)

        obs = torch.cat((obs_in, obs_add_in), dim=1)

        if weights_rec is not None:
            weights_rec = weights_rec.unsqueeze(-1)
            w_idx = torch.tensor([3]).to(self.device)
            pe_assets_init = self.positional_encoding(
                w_idx).view(1, 1, self.d_model)
            pe_assets_init = pe_assets_init.repeat(batch_size,
                self.asset_num, 1)
            pe_assets = pe_assets_init * weights_rec
        else:
            w_idx = torch.tensor([2]).to(self.device)
            pe_assets = self.positional_encoding(
                w_idx).view(1, 1, self.d_model)
            pe_assets = pe_assets.repeat(batch_size, self.asset_num, 1)

        pe = torch.cat((pe_obs, pe_assets), dim=1)

        assets = self.asset_embeds(
            torch.arange(self.asset_num).to(self.device))
        assets = assets.unsqueeze(0)
        assets = assets.repeat(batch_size, 1, 1)

        inputs = torch.cat((obs, assets), dim=1)

        attn_in = inputs + pe

        outputs = self.attn(attn_in, enc_time_mask,
                        enc_key_padding_mask)

        out_preds = self.w_allocator(outputs[:, idx_init:])
        weights = out_preds.squeeze(-1).softmax(dim=-1)

        return weights, outputs