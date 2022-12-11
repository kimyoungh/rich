"""
    Module for Trading Transformer

    @author: Younghyun Kim
    Created on 2022.11.19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mapping import Mapping
from layers.transformer import TransformerEnc
from models.cfg.trading_transformer_config\
    import TRADING_TRANSFORMER_CONFIG


class TradingTransformer(nn.Module):
    """
        Investing Decision Transformer
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
            config = TRADING_TRANSFORMER_CONFIG
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
        self.K = config['K']
        self.action_num = len(self.actions)
        self.value_embeds_nlayers = config['value_embeds_nlayers']
        self.reward_embeds_nlayers = config['reward_embeds_nlayers']
        self.asset_map_nlayers = config['asset_map_nlayers']
        self.value_map_nlayers = config['value_map_nlayers']
        self.action_map_nlayers = config['action_map_nlayers']
        self.reward_map_nlayers = config['reward_map_nlayers']

        # Asset Embeddings
        self.asset_embeds = nn.Sequential(
            nn.Embedding(len(self.assets), self.d_model),
            nn.Dropout(self.dropout))

        # Value Embeddings
        self.value_embeds = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.Tanh(),
            Mapping(self.d_model, self.d_model,
                    self.value_embeds_nlayers,
                    'first', self.slope,
                    self.dropout, True))

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
                    'first', self.slope,
                    self.dropout, True))

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

        # Value Prediction
        self.value_preds = Mapping(self.d_model, 1,
                                self.action_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Action Prediction
        self.action_preds = Mapping(self.d_model, self.action_num,
                                self.action_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Reward Prediction
        self.reward_preds = Mapping(self.d_model, 1,
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

    @torch.no_grad()
    def calc_values(self, values, K=None):
        """
            calculate value by expert_preds(inference time)

            Args:
                values: value predictions by dt
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, 1)
                K: Expert Constant
            Returns:
                value_preds: value prediction with experts
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len, 1)
        """
        if K is None:
            K = self.K
        value_preds = (values + K) / 2.

        return value_preds

    @torch.no_grad()
    def calc_preds(self, logits, sampling=False):
        """
            calculate action and rewards(inference time)

            Args:
                logits: prediction logits by dt
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, dim)
            Returns:
                preds: prediction index
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
        """
        batch_size, seq_len, dim = logits.shape

        logits = logits.softmax(dim=-1).view(
            batch_size*seq_len, dim)

        if sampling:
            preds = torch.multinomial(logits, 1)
        else:
            preds = logits.argmax(-1)

        preds = preds.view(batch_size, seq_len)

        return preds

    def forward(self, assets_in, obs_in, values_in,
                actions_in, rewards_in,
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
                values_in: value index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                rewards_in: reward index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
            Returns:
                value_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len)
                action_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num)
                reward_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
        """
        batch_size, seq_len, _ = obs_in.shape

        assert values_in.shape[0] == batch_size
        assert actions_in.shape[0] == batch_size
        assert rewards_in.shape[0] == batch_size
        assert values_in.shape[1] == actions_in.shape[1]\
            == rewards_in.shape[1] == seq_len

        assets_in = self.asset_embeds(assets_in.unsqueeze(1))

        seq_rng = torch.arange(1, seq_len+1).to(self.device)

        obs_cls = self.obs_embeds(obs_in)

        time_asset = self.time_embeds(
            torch.tensor([0]).to(self.device)).view(1, 1, -1)
        time_asset = time_asset.repeat(batch_size, 1, 1)
        time_embeds = self.time_embeds(seq_rng).view(1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1)

        value_embeds = self.value_embeds(values_in.unsqueeze(-1))
        action_embeds = self.action_embeds(actions_in)
        reward_embeds = self.reward_embeds(rewards_in.unsqueeze(-1))

        assets = assets_in + time_asset
        obs = obs_cls + time_embeds
        values = value_embeds + time_embeds
        actions = action_embeds + time_embeds
        rewards = reward_embeds + time_embeds

        inputs_in = torch.stack((obs, values, actions, rewards),
            dim=1).permute(0, 2, 1, 3).contiguous().view(
                batch_size, seq_len * 4, -1)

        inputs = torch.cat((assets, inputs_in), dim=1)

        outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)
        outputs = outs[:, 1:].view(
            batch_size, seq_len, 4, -1).permute(
                0, 2, 1, 3).contiguous()

        value_preds = self.value_preds(outputs[:, 0]).squeeze(-1)
        action_preds = self.action_preds(outputs[:, 1])
        reward_preds = self.reward_preds(outputs[:, 2]).squeeze(-1)

        return value_preds, action_preds, reward_preds, outputs

    @torch.no_grad()
    def inference(self, assets_in, obs_in, values_in=None,
                actions_in=None, rewards_in=None, K=None,
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
                values_in: value index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                rewards_in: reward index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
            Returns:
                value_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len)
                action_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
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

        if values_in is not None and actions_in is not None and\
            rewards_in is not None:
            assert seq_len == values_in.shape[1]
            assert seq_len == (actions_in.shape[1] + 1)
            assert seq_len == (rewards_in.shape[1] + 1)

            value_embeds = self.value_embeds(values_in.unsqueeze(-1))
            action_embeds = self.action_embeds(actions_in)
            reward_embeds = self.reward_embeds(rewards_in.unsqueeze(-1))

            obs_cls = self.obs_embeds(obs_in)
            obs = obs_cls + time_embeds
            values = value_embeds + time_embeds
            actions = action_embeds + time_embeds[:, :-1]
            rewards = reward_embeds + time_embeds[:, :-1]

            inputs_in = torch.stack(
                (obs[:, :-1], values[:, :-1], actions, rewards),
                dim=1).permute(0, 2, 1, 3).contiguous().view(
                    batch_size, (seq_len-1) * 4, -1)

            inputs = torch.cat(
                (assets, inputs_in, obs[:, -1].unsqueeze(1),
                values[:, -1].unsqueeze(1)), dim=1)

            outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

            action_p = self.action_preds(outs[:, -1]).argmax(-1)

            action_preds = torch.cat(
                (actions_in, action_p.unsqueeze(1)), dim=1)

            return None, action_preds, outs
        else:
            assert seq_len == 1

            obs_cls = self.obs_embeds(obs_in)
            obs = obs_cls + time_embeds

            inputs = torch.cat((assets, obs), dim=1)

            outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

            outputs = outs[:, 1:].view(batch_size, 1, seq_len, -1)

            value_preds = self.calc_values(
                self.value_preds(outputs[:, 0]), K=K)

            value_embeds = self.value_embeds(value_preds)
            values = value_embeds + time_embeds

            inputs = torch.cat((assets, obs, values), dim=1)

            outs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

            outputs = outs[:, 1:].view(batch_size, 2, seq_len, -1)

            action_preds = self.action_preds(outputs[:, 1]).argmax(-1)

            return value_preds.squeeze(-1), action_preds, outs