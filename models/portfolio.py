"""
    Portfolio Models

    @author: Younghyun Kim
    Created: 2022.09.03
"""
import torch
import torch.nn as nn

from layers.mapping import Mapping
from layers.transformer import TransformerEnc

from models.cfg.ipa_config import IPA_CONFIG


class InvestingPortfolioAllocator(nn.Module):
    """
        Investing Portfolio Allocator Class
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
            config = IPA_CONFIG

        self.config = config

        self.factor_num = config['factor_num']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.port_type_num = config['port_type_num']
        self.stock_embeds_map_nlayers =\
            config['stock_embeds_map_nlayers']
        self.w_allocator_nlayers = config['w_allocator_nlayers']

        # Positional Encoding
        # 0: PE Port
        # 1: PE Stock
        self.positional_encoding = nn.Embedding(2, self.d_model)

        # Port Types Encoding
        self.port_types_embeds = nn.Embedding(
            self.port_type_num, self.d_model)

        # Stock Embedding
        self.stock_embeds = Mapping(self.factor_num, self.d_model,
                                    self.stock_embeds_map_nlayers,
                                    'first', self.slope,
                                    self.dropout, True)

        # Transformer Encoder for Allocation
        self.attn = TransformerEnc(self.d_model, self.nhead,
                                self.nlayers, self.d_model * 2,
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

    def forward(self, stocks_in, port_type_idx,
                enc_time_mask=False, enc_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: multifactor scores data for observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num, factor_num)
                port_type_idx: portfolio strategy type index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
            Returns:
                weights: portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num)
                outputs: transformer outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num, d_model)
        """
        batch_size, stock_num, _ = stocks_in.shape

        # Port Idx Info
        port_types = self.port_types_embeds(port_type_idx).unsqueeze(1)

        pe_port = self.positional_encoding(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
        pe_port = pe_port.repeat(batch_size, 1, 1)

        ports = port_types + pe_port

        # Stock Encodings
        stocks_enc = self.stock_embeds(stocks_in)

        pe_stocks = self.positional_encoding(
            torch.tensor([1]).to(self.device)).view(1, 1, self.d_model)
        pe_stocks = pe_stocks.repeat(batch_size, stock_num, 1)

        stocks = stocks_enc + pe_stocks

        inputs = torch.cat((ports, stocks), dim=1)

        outputs = self.attn(inputs,
                            enc_time_mask, enc_key_padding_mask)

        out_preds = self.w_allocator(outputs[:, 1:])
        weights = out_preds.squeeze(-1).softmax(dim=-1)

        return weights, outputs