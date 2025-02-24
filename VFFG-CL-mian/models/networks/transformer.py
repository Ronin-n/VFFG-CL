# import torch.nn as nn
#
# class Transformer(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.AE_input_dim, nhead=opt.nhead, dropout=opt.encoder_dropout)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.n_blocks)
#         decoder_layer = nn.TransformerDecoderLayer(d_model=self.AE_input_dim, nhead=opt.nhead, dropout=opt.decoder_dropout)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=opt.n_blocks)
#
#     def forward(self, x, fusion_feat_miss):
#         latent = self.encoder(x)
#         out = self.decoder(fusion_feat_miss, latent)
#         return out, latent

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        return x + self.pe[:seq_len, :].unsqueeze(0).to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        attn_output, _ = self.attn(x, x, x)
        return attn_output.transpose(0, 1)  # (batch_size, seq_len, d_model)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=512):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForwardNetwork(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.ln1(x + attn_out)  # Residual connection
        ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)  # Residual connection


class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=8, num_layers=3, emb_size=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_size, nhead) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(emb_size, emb_size)  # Adjust to output emb_size

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, emb_size)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.output_layer(x.mean(dim=1))  # Global average pooling


