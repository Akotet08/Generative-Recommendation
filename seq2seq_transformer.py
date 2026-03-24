"""
Transformer-based seq2seq model for generative retrieval.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    """Paper-aligned encoder-decoder Transformer for Semantic-ID generation."""

    def __init__(
        self,
        input_dim,
        output_dim,
        token_dim=128,
        nhead=6,
        head_dim=64,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=256,
    ):
        super().__init__()
        self.d_model = nhead * head_dim

        # Token embeddings (paper-style low-dim token representation).
        self.input_embedding = nn.Embedding(input_dim, token_dim)
        self.output_embedding = nn.Embedding(output_dim, token_dim)

        self.src_projection = nn.Linear(token_dim, self.d_model)
        self.tgt_projection = nn.Linear(token_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.fc_out = nn.Linear(self.d_model, output_dim)

    def _generate_causal_mask(self, tgt_len, device):
        """Upper triangular mask for autoregressive decoding."""
        return torch.triu(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            src: Source token IDs [batch_size, src_len]
            tgt: Decoder input token IDs [batch_size, tgt_len]
            *_key_padding_mask: Optional bool masks (True means pad position)

        Returns:
            logits: [batch_size, tgt_len, output_dim]
        """
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.bool()

        src_embedded = self.src_projection(self.input_embedding(src))
        src_embedded = src_embedded * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)

        tgt_embedded = self.tgt_projection(self.output_embedding(tgt))
        tgt_embedded = tgt_embedded * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        memory = self.transformer_encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.fc_out(output)

    def encode(self, src, src_key_padding_mask=None):
        """Encode source tokens into memory representations.

        Args:
            src: Source token IDs [batch_size, src_len]
            src_key_padding_mask: Optional bool mask (True means pad position)

        Returns:
            memory: [batch_size, src_len, d_model]
        """
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()
        src_embedded = self.src_projection(self.input_embedding(src))
        src_embedded = src_embedded * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        return self.transformer_encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Decode target tokens using pre-computed encoder memory.

        Args:
            tgt: Decoder input token IDs [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_key_padding_mask: Optional bool mask (True means pad position)
            memory_key_padding_mask: Optional bool mask (True means pad position)

        Returns:
            logits: [batch_size, tgt_len, output_dim]
        """
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.bool()
        tgt_embedded = self.tgt_projection(self.output_embedding(tgt))
        tgt_embedded = tgt_embedded * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)
