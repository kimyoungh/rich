"""
    Target Finish Utils for TradingDT

    @author: Younghyun Kim
    Created: 2023.04.30
"""
import torch
import torch.nn as nn

def direction_tf(action_preds_target, returns, coeff):
    """
        Direction based Target Finish

        Args:
            action_preds_target: action prediction logits for target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period, action_num)
            returns: return series for underlying asset
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            coeff: coefficient  for turnover_tf
                * dtype: torch.FloatTensor
                * shape: (1)
    """
    loss_fn = nn.CrossEntropyLoss()
    batch_size, _, action_num = action_preds_target.shape

    updowns = (returns < 0).type(torch.long).view(-1)
    loss_d = loss_fn(action_preds_target.view(-1, action_num),
                    updowns)

    loss_direc = coeff * loss_d

    return loss_direc

def avg_return_tf(actions_target, actions_comp,
                rets_target, rets_comp, coeff):
    """
        Average Return based Target Finish

        Args:
            actions_target: actions of target model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            actions_comp: actions of comparative model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            rets_target: returns of target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            rets_comp: returns of comparative model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            coeff: coefficient for avg_return_tf
                * dtype: torch.FloatTensor
                * shape: (1)
        Returns:
            loss_r: loss for avg_return_tf
                * dtype: torch.FloatTensor
                * shape: (1,)
    """
    batch_size = actions_target.shape[0]
    coefficient = coeff.view(1, 1).repeat(batch_size, 1)

    returns_target = actions_target * rets_target
    returns_comp = actions_comp * rets_comp

    loss_r = ((-coefficient) * torch.log(
        (returns_target - returns_comp).sigmoid()).mean(dim=1)).mean()

    return loss_r

def cagr_tf(actions_target, actions_comp,
            rets_target, rets_comp, coeff):
    """
        CAGR based Target Finish

        Args:
            actions_target: actions of target model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            actions_comp: actions of comparative model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            rets_target: returns of target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            rets_comp: returns of comparative model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            coeff: coefficient for avg_return_tf
                * dtype: torch.FloatTensor
                * shape: (1)
        Returns:
            loss_c: loss for cagr_tf
                * dtype: torch.FloatTensor
                * shape: (1,)
    """
    batch_size, trading_period = actions_target.shape
    coefficient = coeff.view(1, 1).repeat(batch_size, 1)

    returns_target = actions_target * rets_target
    returns_comp = actions_comp * rets_comp

    crets_target = (1 + returns_target).cumprod(dim=1)
    crets_comp = (1 + returns_comp).cumprod(dim=1)

    cagr_target = crets_target[:, -1] ** (1. / float(trading_period))
    cagr_comp = crets_comp[:, -1] ** (1. / float(trading_period))

    loss_c = ((-coefficient) * torch.log(
        (cagr_target - cagr_comp).sigmoid())).mean()

    return loss_c

def turnover_tf(action_preds_target, action_preds_comp, coeff):
    """
        Target Finish based on Turnover Loss

        Args:
            action_preds_target: action prediction logits for target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period, action_num)
            action_preds_comp: action prediction logits for comparative model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period, action_num)
            coeff: coefficient  for turnover_tf
                * dtype: torch.FloatTensor
                * shape: (1)
        Returns:
            loss_t: loss for turnover_tf
                * dtype: torch.FloatTensor
                * shape: (1,)
    """
    loss_kld = nn.KLDivLoss(reduction='none')

    action_preds_fwd = action_preds_target[:, 1:].contiguous()
    action_preds_bak = action_preds_target[:, :-1].detach().contiguous()

    action_preds_cfwd = action_preds_comp[:, 1:].contiguous()
    action_preds_cbak = action_preds_comp[:, :-1].contiguous()

    batch_size, _, action_num = action_preds_fwd.shape

    coefficient = coeff.view(1, 1).repeat(batch_size, 1)

    loss_k = loss_kld(
        action_preds_fwd.view(-1, action_num).log_softmax(-1),
        action_preds_bak.view(-1, action_num).softmax(-1))
    
    loss_ck = loss_kld(
        action_preds_cfwd.view(-1, action_num).log_softmax(-1),
        action_preds_cbak.view(-1, action_num).softmax(-1))

    loss_k = loss_k.view(batch_size, -1, action_num).sum(dim=-1)
    loss_ck = loss_ck.view(batch_size, -1, action_num).sum(dim=-1)

    loss_t = (coefficient * torch.log(
        (loss_k - loss_ck).sigmoid()).mean(dim=1)).mean()

    return loss_t

def volatility_tf(actions_target, actions_comp,
                rets_target, rets_comp, coeff):
    """
        Target Finish based on Volatility Loss

        Args:
            actions_target: actions of target model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            actions_comp: actions of comparative model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            rets_target: returns of target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            rets_comp: returns of comparative model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            coeff: coefficient for avg_return_tf
                * dtype: torch.FloatTensor
                * shape: (1)
        Returns:
            loss_v: loss for volatility_tf
                * dtype: torch.FloatTensor
                * shape: (1,)
    """
    batch_size, _ = actions_target.shape
    coefficient = coeff.view(1, 1).repeat(batch_size, 1)

    returns_target = actions_target * rets_target
    returns_comp = actions_comp * rets_comp

    vol_target = returns_target.std(dim=-1)
    vol_comp = returns_comp.std(dim=-1)

    loss_v = (coefficient * torch.log(
        (vol_target - vol_comp).sigmoid())).mean()

    return loss_v

def mdd_tf(actions_target, actions_comp,
        rets_target, rets_comp, coeff, eps=1e-6):
    """
        Target Finish based on MDD Loss

        Args:
            actions_target: actions of target model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            actions_comp: actions of comparative model
                * dtype: torch.LongTensor
                * shape: (batch_size, trading_period)
            rets_target: returns of target model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            rets_comp: returns of comparative model
                * dtype: torch.FloatTensor
                * shape: (batch_size, trading_period)
            coeff: coefficient for avg_return_tf
                * dtype: torch.FloatTensor
                * shape: (1)
            epsilon: epsilon
                * default: 1e-6
        Returns:
            loss_m: loss for mdd_tf
                * dtype: torch.FloatTensor
                * shape: (1,)
    """
    batch_size, _ = actions_target.shape
    coefficient = coeff.view(1, 1).repeat(batch_size, 1)

    returns_target = actions_target * rets_target
    returns_comp = actions_comp * rets_comp

    crets_target = (1 + returns_target).cumprod(dim=1)
    crets_comp = (1 + returns_comp).cumprod(dim=1)

    cmax_target, _ = crets_target.cummax(dim=1)
    cmax_comp, _ = crets_comp.cummax(dim=1)

    dd_target = -(crets_target - cmax_target) / (cmax_target + eps)
    dd_comp = -(crets_comp - cmax_comp) / (cmax_comp + eps)

    mdd_target, _ = dd_target.max(dim=1)
    mdd_comp, _ = dd_comp.max(dim=1)

    loss_m = (coefficient * torch.log(
        (mdd_target - mdd_comp).sigmoid())).mean()

    return loss_m