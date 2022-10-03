"""
    Beam Trader

    @author: Younghyun Kim
    Created: 2022.09.25
"""
import torch
import torch.nn as nn

from layers.mapping import Mapping
from layers.transformer import TransformerEnc

from models.cfg.beam_trader_config import BEAM_TRADER_CONFIG


class BeamTrader(nn.Module):
    """
        Beam Trader
    """
    def __init__(self, config: dict = None):
        """
            Initialization
        """
        super().__init__()

        if config is None:
            config = BEAM_TRADER_CONFIG

        self.config = config
        self.factor_num = config['factor_num']
        self.action_num = config['action_num']
        self.max_len = config['max_len']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.obs_embeds_map_nlayers =\
            config['obs_embeds_map_nlayers']
        self.action_preds_map_nlayers =\
            config['action_preds_map_nlayers']

        # Positional Encoding
        self.positional_encoding = nn.Embedding(
            self.max_len, self.d_model)

        # Action_init
        self.action_init = nn.Embedding(1, self.d_model)

        # Action Encoding
        self.action_embeds = nn.Embedding(self.action_num, self.d_model)

        # Action Predictor
        self.action_preds = Mapping(self.d_model, self.action_num,
                                self.action_preds_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Obs Embedding
        self.obs_embeds = Mapping(self.factor_num, self.d_model,
                                self.obs_embeds_map_nlayers,
                                'first', self.slope,
                                self.dropout, True)

        # Transformer
        self.attn = TransformerEnc(self.d_model, self.nhead,
                                self.nlayers, self.d_model * 2,
                                self.dropout, self.activation, True)

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

    def forward(self, obs_in, actions_in, action_init=None,
            enc_time_mask=True, enc_key_padding_mask=None):
        """
            Inference

            Args:
                obs_in: multifactor scores data for observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, factor_num)
                actions_in: action sequence
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                action_init: initial action
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, 1)
                    * default: None
            Return:
                logits: logits from transformer
                    * dtype: Torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num)
        """
        batch_size, seq_len = actions_in.shape
        traj_len = seq_len + 2

        if action_init is None:
            action_init = self.action_init(
                torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
            action_init = action_init.repeat(batch_size, 1, 1)
        else:
            action_init = self.action_embeds(action_init)

        # Obs Embedding
        obs_in = self.obs_embeds(obs_in)
        obs_embeds = obs_in.mean(dim=1, keepdims=True)

        # Action Embeddings
        actions_in = self.action_embeds(actions_in)

        # Positional Encoding
        pe = self.positional_encoding(
            torch.arange(traj_len).to(self.device)).view(
                1, traj_len, self.d_model)
        pe = pe.repeat(batch_size, 1, 1)

        attn_in = torch.cat(
            (action_init, obs_embeds, actions_in), dim=1)

        outputs = self.attn(attn_in, seq_mask=enc_time_mask,
                            src_key_padding_mask=enc_key_padding_mask)

        logits = self.action_preds(outputs[:, 1:])

        return logits