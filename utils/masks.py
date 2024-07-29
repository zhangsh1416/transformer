import torch

def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.float()
