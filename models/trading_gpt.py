"""
    Module for Trading GPT

    @author: Younghyun Kim
    Created on 2023.03.19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mapping import Mapping
from layers.transformer import TransformerEnc
from models.cfg.trading_gpt_config import TRADING_GPT_CONFIG


class TradingGPT(nn.Module):
    """
        TradingGPT
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
            config = TRADING_GPT_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.d_model = config['d_model']
        self.dim_ff = config['dim_ff']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.max_len = config['max_len']
        self.actions = config['actions']
        self.assets = config['assets']
        self.action_num = len(self.actions)
        self.reward_embeds_nlayers = config['reward_embeds_nlayers']
        self.obs_map_nlayers = config['obs_map_nlayers']
        self.action_map_nlayers = config['action_map_nlayers']
        self.reward_map_nlayers = config['reward_map_nlayers']

        # Asset Embeddings
        self.asset_embeds = nn.Sequential(
            nn.Embedding(len(self.assets), self.d_model),
            nn.Dropout(self.dropout))

        # Action Embeddings
        self.action_embeds = nn.Sequential(
            nn.Embedding(self.action_num, self.d_model),
            nn.Dropout(self.dropout))

        # Reward Embeddings
        self.reward_embeds = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.Tanh(),
            Mapping(self.d_model, self.d_model,
                    self.reward_embeds_nlayers,
                    'first', self.slope, self.dropout, True))

        # Time Encodings
        self.time_embeds = nn.Sequential(
            nn.Embedding(self.max_len, self.d_model),
            nn.Dropout(self.dropout))

        # Observation Data Encoder
        self.obs_embeds = Mapping(self.factor_num, self.d_model,
                            self.obs_map_nlayers,
                            'first', self.slope, self.dropout,
                            True)

        # Decision Transformer
        self.attn = TransformerEnc(self.d_model, self.nhead,
                                self.nlayers,
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

    def forward(self, assets_in, obs_in,
                actions_in=None, rewards_in=None,
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
                    * shape: (batch_size, act_len)
                rewards_in: reward index series
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, rew_len)
            Returns:
                action_preds
                    * dtype: torch.FloatTensor
                    if seq_len == act_len:
                        * shape: (batch_size, seq_len, action_num)
                    elif seq_len == (act_len + 1):
                        * shape: (batch_size, 1, 2)
                    elif actions_in is None:
                        * shape: (batch_size, 1, 2)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
        """
        batch_size, seq_len, _ = obs_in.shape

        # Asset Embeddings
        assets_in = self.asset_embeds(assets_in.unsqueeze(1))

        time_asset = self.time_embeds(
            torch.tensor([0]).to(self.device)).view(1, 1, -1)
        time_asset = time_asset.repeat(batch_size, 1, 1)

        assets = assets_in + time_asset

        # Time Encodings
        seq_rng = torch.arange(1, seq_len+1).to(self.device)

        time_embeds = self.time_embeds(seq_rng).view(1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1)

        # Observation Embeddings
        obs_embeds = self.obs_embeds(obs_in)
        obs = obs_embeds + time_embeds

        if actions_in is not None and rewards_in is not None:
            _, act_len = actions_in.shape
            _, rew_len = rewards_in.shape

            assert act_len == rew_len

            action_embeds = self.action_embeds(actions_in)
            reward_embeds = self.reward_embeds(rewards_in.unsqueeze(-1))

            if seq_len == act_len:
                actions = action_embeds + time_embeds
                rewards = reward_embeds + time_embeds

                inputs_in = torch.stack((obs, actions, rewards),
                dim=1).permute(0, 2, 1, 3).contiguous().view(
                    batch_size, seq_len * 3, -1)

                inputs = torch.cat((assets, inputs_in), dim=1)

                outs = self.attn(inputs, seq_mask=seq_mask,
                                src_key_padding_mask=src_key_padding_mask)
                outputs = outs[:, 1:].view(
                    batch_size, seq_len, 3, -1).permute(
                        0, 2, 1, 3).contiguous()

                action_preds = self.action_preds(outputs[:, 0])
            elif seq_len == (act_len + 1):
                actions = action_embeds + time_embeds[:, :-1]
                rewards = reward_embeds + time_embeds[:, :-1]

                inputs_in = torch.stack(
                    (obs[:, :-1], actions, rewards),
                    dim=1).permute(0, 2, 1, 3).contiguous().view(
                        batch_size, (seq_len-1) * 3, -1)
                inputs = torch.cat(
                    (assets, inputs_in, obs[:, -1].unsqueeze(1)), dim=1)

                outs = self.attn(inputs, seq_mask=seq_mask,
                                src_key_padding_mask=src_key_padding_mask)
                action_preds = self.action_preds(
                    outs[:, -1]).view(batch_size, 1, -1)
        else:
            assert seq_len == 1

            inputs = torch.cat((assets, obs), dim=1)

            outs = self.attn(inputs, seq_mask=seq_mask,
                            src_key_padding_mask=src_key_padding_mask)
            outputs = outs[:, 1:].view(batch_size, 1, seq_len, -1)

            action_preds = self.action_preds(
                outputs[:, 0]).view(batch_size, 1, -1)

        return action_preds, outs