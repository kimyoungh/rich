"""
    Fundamental NN Layers - Transformer

    @author: Younghyun Kim
    Created: 2022.09.03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F




class TransformerEnc(nn.Module):
    """
        Transformer Encoder
    """
    def __init__(self, d_model, nhead=4, nlayers=6,
                dim_feedforward=2048, dropout=0.1,
                activation='relu', batch_first=True):
        """
            batch_first: batch dimension(default: False)
                * True: input shape -> (batch_size, seq_len, emb_size)
                * False: input shape -> (seq_len, batch_size, emb_size)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_first = batch_first

        self.enc_layer =\
            nn.TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward,
                                    dropout, activation,
                                    batch_first=batch_first)

        self.attn_enc = nn.TransformerEncoder(self.enc_layer,
                                            num_layers=nlayers)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_square_subsequent_mask(self, seq_len):
        """
            generate Square Subsequent Mask
        """
        mask = (torch.triu(torch.ones((seq_len, seq_len),
        device=self.device)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0,
            float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, x_in, seq_mask=False, src_key_padding_mask=None):

        if self.batch_first:
            seq_len = x_in.shape[1]
        else:
            seq_len = x_in.shape[0]

        if seq_mask:
            mask = self.generate_square_subsequent_mask(seq_len)
        else:
            mask = None

        out_embs = self.attn_enc(x_in, mask=mask,
                            src_key_padding_mask=src_key_padding_mask)

        return out_embs


class TransformerDec(nn.Module):
    """
        Transformer Decoder
    """
    def __init__(self, d_model, nhead=4, nlayers=6,
                dim_feedforward=2048, dropout=0.1,
                activation='relu', batch_first=True):
        """
            batch_first: batch dimension(default: False)
                * True: input shape -> (batch_size, seq_len, emb_size)
                * False: input shape -> (seq_len, batch_size, emb_size)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_first = batch_first

        self.dec_layer =\
            nn.TransformerDecoderLayer(d_model, nhead,
                                    dim_feedforward, dropout,
                                    activation, batch_first=batch_first)

        self.attn_dec = nn.TransformerDecoder(self.dec_layer,
                                            num_layers=nlayers)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_square_subsequent_mask(self, seq_len):
        """
            generate Square Subsequent Mask
        """
        mask = (torch.triu(torch.ones((seq_len, seq_len),
        device=self.device)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0,
            float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, tgt, enc_memory,
                tgt_mask=False, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        if self.batch_first:
            seq_len = tgt.shape[1]
        else:
            seq_len = tgt.shape[0]

        if tgt_mask:
            mask = self.generate_square_subsequent_mask(seq_len)
        else:
            mask = None

        out_embs =\
            self.attn_dec(tgt, enc_memory, tgt_mask=mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

        return out_embs