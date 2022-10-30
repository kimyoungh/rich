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

        # Additional Obs Embedding
        self.obs_add_embeds = Mapping(self.factor_num, self.d_model,
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

    def forward(self, obs_in, obs_add_in=None,
                actions_in=None, action_init=None,
                enc_time_mask=True, enc_key_padding_mask=None):
        """
            Inference

            Args:
                obs_in: multifactor scores data for observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, factor_num)
                obs_add_in: additional multifactor scores data for observations
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
        traj_len = seq_len + 3

        if actions_in is None:
            actions_in = self.action_embeds(
                torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
            actions_in = actions_in.repeat(batch_size, 1, 1)

        if action_init is None:
            action_init = self.action_init(
                torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
            action_init = action_init.repeat(batch_size, 1, 1)
        else:
            action_init = self.action_embeds(action_init)

        # Obs Embedding
        obs_in = self.obs_embeds(obs_in)

        if obs_add_in is not None:
            obs_add_in = self.obs_embeds(obs_add_in)
        else:
            obs_add_in = torch.zeros_like(obs_in).to(self.device)

        obs_embeds = torch.cat(
            (obs_in.mean(dim=1, keepdims=True),
            obs_add_in.mean(dim=1, keepdims=True)), dim=1)

        # Action Embeddings
        actions_in = self.action_embeds(actions_in)

        # Positional Encoding
        pe = self.positional_encoding(
            torch.arange(traj_len).to(self.device)).view(
                1, traj_len, self.d_model)
        pe = pe.repeat(batch_size, 1, 1)

        attn_in = torch.cat(
            (obs_embeds, action_init, actions_in), dim=1)

        outputs = self.attn(attn_in + pe, seq_mask=enc_time_mask,
                            src_key_padding_mask=enc_key_padding_mask)

        logits = self.action_preds(outputs[:, 2:])

        return logits

    @torch.no_grad()
    def beam_plan(self, obs_in, obs_add_in=None, action_init=None,
                seq_len=10, beam_width=10, n_expand=10,
                discount=0.99):
        """
            Beam Plan

            Returns:
                best_action: best action for now
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
                best_actions: best action series from now
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
        """
        self.eval()

        obs_in = obs_in.to(self.device)

        if obs_add_in is not None:
            obs_add_in = obs_add_in.to(self.device)

        batch_size = len(obs_in)
        actions = torch.zeros(
            (batch_size, beam_width, seq_len)).to(self.device)

        actions = actions.type(torch.long)
        discounts = discount ** torch.arange(seq_len).to(self.device)

        length = seq_len
        t = 0
        topk = beam_width

        while length > 0:
            if t == 0:
                width = beam_width
            else:
                actions = actions.repeat(1, n_expand, 1)
                width = actions.shape[1]

            o_in = obs_in.unsqueeze(1)
            o_in = o_in.repeat(1, width, 1, 1)
            o_in = o_in.view(-1, obs_in.shape[-2],
                                obs_in.shape[-1])

            if obs_add_in is not None:
                o_add_in = obs_add_in.unsqueeze(1)
                o_add_in = o_add_in.repeat(1, width, 1, 1)
                o_add_in = o_add_in.view(-1,
                        obs_add_in.shape[-2],
                        obs_add_in.shape[-1]).contiguous()

            actions_in = actions[:, :, :t+1]

            if action_init is not None:
                act_init = action_init.unsqueeze(1)
                act_init = act_init.repeat(1, width, 1)
                act_init = act_init.view(-1, 1).contiguous()
            else:
                act_init = action_init

            logits = self.forward(o_in, o_add_in,
                actions_in.view(-1, t+1).contiguous(),
                act_init)

            logits = logits.view(batch_size, width, t+2, -1)
            logits = logits[:, :, :-1].contiguous()

            acts = self.sample_actions(logits)
            log_probs = logits.log_softmax(-1)
            logps = log_probs.gather(3, acts.unsqueeze(-1)).squeeze(-1)

            cumulative_logp = (
                logps * discounts[:t+1]).sum(dim=-1)

            cumulative_logp, inds = torch.topk(cumulative_logp,
                                            topk, dim=1)

            actions[:, :, :t+1] = acts
            actions = actions.gather(1,
                inds.unsqueeze(-1).repeat_interleave(seq_len, -1))

            t += 1
            length -= 1

        argmax = cumulative_logp.argmax(dim=1)
        best_actions = actions.gather(1,
            argmax.view(-1, 1, 1).repeat_interleave(seq_len, -1))
        best_actions = best_actions.squeeze(1)
        best_action = best_actions[:, 0]

        return best_action, best_actions

    @torch.no_grad()
    def sample_actions(self, logits):
        """
            sample actions
        """
        probs = logits.softmax(-1).view(-1, logits.shape[-1]).contiguous()
        acts = torch.multinomial(probs, num_samples=1,
                                replacement=True)
        acts = acts.view(logits.shape[0], logits.shape[1],
                        logits.shape[2])

        return acts