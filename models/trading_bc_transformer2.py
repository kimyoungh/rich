"""
    Module for Trading BC Transformer 2

    @author: Younghyun Kim
    Created on 2022.12.18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mapping import Mapping
from layers.transformer import TransformerEnc
from models.cfg.trading_bc_transformer2_config\
    import TRADING_BC_TRANSFORMER2_CONFIG


class TradingBCTransformer2(nn.Module):
    """
        Trading BC Transformer
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
            config = TRADING_BC_TRANSFORMER2_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.d_model = config['d_model']
        self.dim_ff = config['dim_ff']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.nhead = config['tt_nhead']
        self.nlayers = config['tt_nlayers']
        self.activation = config['activation']
        self.max_len = config['max_len']
        self.actions = config['actions']
        self.assets = config['assets']
        self.action_num = len(self.actions)
        self.asset_map_nlayers = config['asset_map_nlayers']
        self.action_map_nlayers = config['action_map_nlayers']

        # Asset Embeddings
        self.asset_embeds = nn.Sequential(
            nn.Embedding(len(self.assets), self.d_model),
            nn.Dropout(self.dropout))

        # Action Embeddings
        self.action_embeds = nn.Sequential(
            nn.Embedding(self.action_num, self.d_model),
            nn.Dropout(self.dropout))

        # Time Encoding
        self.time_embeds = nn.Sequential(
            nn.Embedding(self.max_len, self.d_model),
            nn.Dropout(self.dropout))

        # Observation Data Encoder
        self.obs_embeds = Mapping(self.factor_num, self.d_model,
                                self.asset_map_nlayers,
                                'first', self.slope, self.dropout,
                                True)

        # Decision Transformer
        self.dt = TransformerEnc(self.d_model, self.nhead, self.nlayers,
                                self.dim_ff, self.dropout,
                                self.activation, True)

        # Action Prediction
        self.action_preds = Mapping(self.d_model, self.action_num,
                                self.action_map_nlayers,
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

    def forward(self, assets_in, obs_in, actions_in,
                seq_mask=True, src_key_padding_mask=None):
        """
            forward

            Args:
                assets_in: asset index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
                obs_in: observations for Asset Factors
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, factor_num)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
            Returns:
                action_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
        """
        batch_size, seq_len, _ = obs_in.shape

        assert actions_in.shape[0] == batch_size
        assert actions_in.shape[1] == seq_len

        assets_in = self.asset_embeds(assets_in.unsqueeze(1))

        seq_rng = torch.arange(1, seq_len+1).to(self.device)

        obs_cls = self.obs_embeds(obs_in)

        time_asset = self.time_embeds(
            torch.tensor([0]).to(self.device)).view(1, 1, -1)
        time_asset = time_asset.repeat(batch_size, 1, 1)
        time_embeds = self.time_embeds(seq_rng).view(1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1)

        action_embeds = self.action_embeds(actions_in)

        assets = assets_in + time_asset
        obs = obs_cls + time_embeds
        actions = action_embeds + time_embeds

        inputs_in = torch.stack((obs, actions),
            dim=1).permute(0, 2, 1, 3).contiguous().view(
                batch_size, seq_len * 2, -1)

        inputs = torch.cat((assets, inputs_in), dim=1)

        outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)
        outputs = outs[:, 1:].view(
            batch_size, seq_len, 2, -1).permute(
                0, 2, 1, 3).contiguous()

        action_preds = self.action_preds(outputs[:, 0])

        return action_preds, outputs

    @torch.no_grad()
    def inference(self, assets_in, obs_in, actions_in=None,
                seq_mask=True, src_key_padding_mask=None):
        """
            Inference

            Args:
                assets_in: asset index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
                obs_in: observations for Asset Factors
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, factor_num)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len-1)
            Returns:
                action_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
                action_pr
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num)
        """
        batch_size, seq_len, _ = obs_in.shape

        assets_in = self.asset_embeds(assets_in.unsqueeze(1))

        time_asset = self.time_embeds(
            torch.tensor([0]).to(self.device)).view(1, 1, -1)
        time_asset = time_asset.repeat(batch_size, 1, 1)

        assets = assets_in + time_asset

        seq_rng = torch.arange(1, seq_len+1).to(self.device)

        time_embeds = self.time_embeds(seq_rng).view(1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1)

        if actions_in is not None:
            assert seq_len == (actions_in.shape[1] + 1)

            action_embeds = self.action_embeds(actions_in)

            obs_cls = self.obs_embeds(obs_in)
            obs = obs_cls + time_embeds
            actions = action_embeds + time_embeds[:, :-1]

            inputs_in = torch.stack(
                (obs[:, :-1], actions),
                dim=1).permute(0, 2, 1, 3).contiguous().view(
                    batch_size, (seq_len-1) * 2, -1)

            inputs = torch.cat(
                (assets, inputs_in, obs[:, -1].unsqueeze(1)), dim=1)

            outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

            action_pr = self.action_preds(outs[:, -1])
            action_p = action_pr.argmax(-1)

            action_preds = torch.cat(
                (actions_in, action_p.unsqueeze(1)), dim=1)

            return action_preds, outs, action_pr.softmax(dim=-1)
        else:
            assert seq_len == 1

            obs_cls = self.obs_embeds(obs_in)
            obs = obs_cls + time_embeds

            inputs = torch.cat((assets, obs), dim=1)

            outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

            outputs = outs[:, 1:].view(batch_size, 1, seq_len, -1)

            action_pr = self.action_preds(outputs[:, 0])
            action_preds = action_pr.argmax(-1)

            return action_preds, outs, action_pr.softmax(dim=-1)