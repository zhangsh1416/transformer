import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_len, dropout)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, enc_input, dec_input, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(enc_input, enc_padding_mask)
        dec_output, attention_weights = self.decoder(dec_input, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
