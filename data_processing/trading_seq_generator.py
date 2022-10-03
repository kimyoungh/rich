"""
    Trading Sequence Generator for BeamTrader

    @author: Younghyun Kim
    Created: 2022.10.01
"""
import numpy as np
import pandas as pd
import torch
import pdb


class TradingSeqGenerator:
    """
        Trding Sequence Generator for BeamTrader
    """
    def __init__(self, returns_data, seq_len=20,
                trading_fee=0.01,
                beam_width=10, n_expand=20, discount=0.99,
                device='cpu'):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                seq_len: forward sequence length
                trading_fee: trading fee
                    * default: 0.01
                beam_width: beam width
                    * default: 10
                n_expand: number of expansion
                    * default: 20
                discount: discount
                    * default: 0.99
        """
        self.seq_len = seq_len
        self.trading_fee = trading_fee * 2.
        self.beam_width = beam_width
        self.n_expand = n_expand
        self.discount = discount
        self.device = device

        self.data_length, self.stock_num = returns_data.shape

        # Action Index
        ## 0: Cash
        ## 1: Stock 1
        ## 2: Stock 2
        ## ....
        self.actions = np.arange(self.stock_num+1)
        self.action_num = self.actions.shape[0]

        cash = pd.DataFrame(np.zeros((self.data_length, 1)),
                            index=returns_data.index,
                            columns=['Cash'])

        returns_data = pd.concat((cash, returns_data), axis=1)

        self.returns_data = returns_data

        probs = torch.ones(self.action_num) / float(self.action_num)
        self.probs = probs.to(self.device)

    def generate_beam_sequences(self, rebal=False):
        """
            Generate Beam Sequences

            Args:
                rebal: rebalancing initial date or not
            Returns:
                best_seqs
                    * dtype: np.array
                    * shape
                        * rebal: (date_num, action_num, seq_len)
                        * no rebal: (date_num, seq_len)
                best_rews
                    * dtype: np.array
                    * shape
                        * rebal: (date_num, action_num, seq_len)
                        * no rebal: (date_num, seq_len)
                best_vals
                    * dtype: np.array
                    * shape
                        * rebal: (date_num, action_num)
                        * no rebal: (date_num)
                indices
                    * dtype: np.array
                    * shape: (date_num)
        """
        indices = np.arange(self.data_length - self.seq_len - 1)
        if rebal:
            best_seqs = torch.zeros(
                (len(indices), self.action_num, self.seq_len)
                ).type(torch.long).to(self.device)
            best_vals = torch.zeros(
                (len(indices), self.action_num)).to(self.device)
        else:
            best_seqs = torch.zeros(
                (len(indices), self.seq_len)).type(torch.long).to(self.device)
            best_vals = torch.zeros(len(indices)).to(self.device)

        best_seqs, best_rews, best_vals = [], [], []

        for t in range(self.data_length - self.seq_len - 1):
            best_seq, best_rew, best_val = self.beam_search(t, rebal=rebal)
            best_seqs.append(best_seq)
            best_rews.append(best_rew)
            best_vals.append(best_val.view(-1))

        best_seqs = torch.stack(best_seqs, dim=0)
        best_rews = torch.stack(best_rews, dim=0)
        best_vals = torch.stack(best_vals, dim=0)

        best_seqs = best_seqs.cpu().numpy()
        best_rews = best_rews.cpu().numpy()
        best_vals = best_vals.cpu().numpy()

        return best_seqs, best_rews, best_vals, indices

    def beam_search(self, init_t, rebal=False):
        """
            Beam Search

            Args:
                init_t: initial timing
            Return:
                best_sequence: best action sequence
                    * dtype: torch.FloatTensor
                    * shape: (seq_len)
                best_rewards: best reward sequence
                    * dtype: torch.FloatTensor
                    * shape: (seq_len)
                best_value: best value
        """
        assert (init_t + self.seq_len) <= (self.data_length - 2)

        seq_len = self.seq_len
        t = 0

        if rebal:
            returns = self.returns_data.iloc[init_t:init_t+self.seq_len]

            rewards = torch.zeros((
                self.action_num, self.action_num, self.seq_len)
            ).to(self.device)
            actions = torch.zeros_like(rewards).to(self.device)

            actions[:, :, 0] = torch.LongTensor(
                self.actions
                ).to(self.device).unsqueeze(0).repeat(
                    self.action_num, 1).transpose(0, 1)

            rew_init = torch.FloatTensor(
                returns.iloc[0].values.astype(float)).to(self.device)
            rew_init = rew_init.unsqueeze(0).repeat(self.action_num, 1)

            rewards[:, :, 0] = rew_init.transpose(0, 1)

            t += 1
            seq_len -= 1
        else:
            returns = self.returns_data.iloc[init_t+1:init_t+self.seq_len+1]
            rewards = torch.zeros(
                (self.action_num, self.seq_len)).to(self.device)
            actions = torch.zeros_like(rewards).to(self.device)

        actions = actions.type(torch.long)
        discounts = self.discount ** torch.arange(self.seq_len).to(self.device)

        while seq_len > 0:
            if rebal:
                actions = actions.repeat(1, self.n_expand, 1)
                rewards = rewards.repeat(1, self.n_expand, 1)
                actions = actions.view(-1, self.seq_len).contiguous()
                rewards = rewards.view(-1, self.seq_len).contiguous()

                topk = self.beam_width
            elif t > 0:
                actions = actions.repeat(self.n_expand, 1)
                rewards = rewards.repeat(self.n_expand, 1)

                topk = self.beam_width
            else:
                topk = self.action_num

            acts_picked = self.sample_actions(actions.shape[0])
            rew_picked = torch.FloatTensor(
                returns.iloc[
                    t, acts_picked.cpu().numpy()].values.astype(float)
            ).to(self.device)

            actions[:, t] = acts_picked

            if t > 0:
                rew_picked = rew_picked.unsqueeze(1)
                rew_picked[
                    actions[:, t] != actions[:, t-1]] -= self.trading_fee
                rew_picked = rew_picked.view(-1)

            rewards[:, t] = rew_picked

            values = (rewards * discounts).sum(dim=-1)

            if rebal:
                actions = actions.view(
                    self.action_num, -1, self.seq_len).contiguous()
                rewards = rewards.view(
                    self.action_num, -1, self.seq_len).contiguous()

                values = values.view(self.action_num, -1).contiguous()

                values, inds = torch.topk(values, topk, dim=1)
                inds = inds.unsqueeze(-1).repeat_interleave(self.seq_len, -1)
                actions = actions.gather(1, inds)
                rewards = rewards.gather(1, inds)
            else:
                values, inds = torch.topk(values, topk)

                actions = actions[inds]
                rewards = rewards[inds]

            t += 1
            seq_len -= 1

        if rebal:
            argmax = values.argmax(dim=1).view(-1, 1, 1)

            best_sequence = actions.gather(
                1, argmax.repeat_interleave(self.seq_len, -1)).squeeze(1)
            best_rewards = rewards.gather(
                1, argmax.repeat_interleave(self.seq_len, -1)).squeeze(1)
            best_value = values.gather(1, argmax.squeeze(-1))
        else:
            argmax = values.argmax()
            best_sequence = actions[argmax]
            best_rewards = rewards[argmax]
            best_value = values[argmax]

        return best_sequence, best_rewards, best_value

    def sample_actions(self, nums):
        """
            Sample Actions
        """
        probs = self.probs.unsqueeze(0).repeat(nums, 1)

        picked = torch.multinomial(probs, num_samples=1,
                                replacement=True)

        return picked.view(-1)