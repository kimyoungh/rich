"""
    Module for TradingDT
    (Trading Decision Transformer)

    @author: Younghyun Kim
    Created: 2023.04.30
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mapping import Mapping
from layers.transformer import TransformerEnc
from models.cfg.trading_dt_config import TRADING_DT_CONFIG


class TradingDT(nn.Module):
    """
        TradingDT - Trading Decision Transformer
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
        """
        super().__init__()

        if config is None:
            config = TRADING_DT_CONFIG