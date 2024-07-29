import torch
from models.transformer import Transformer
from utils.masks import create_padding_mask, create_look_ahead_mask

# 超参数
num_layers = 4
d_model = 128
num_heads = 8
d_ff = 512
input_vocab_size = 8000
target_vocab_size = 8000
max_seq_len = 100
dropout = 0.1
batch_size = 32

# 模型初始化
model = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout)

# 假设我们有输入和目标张量enc_input, dec_input
enc_input = torch.randint(0, input_vocab_size, (batch_size, max_seq_len))
dec_input = torch.randint(0, target_vocab_size, (batch_size, max_seq_len))

# 测试步骤
enc_padding_mask = create_padding_mask(enc_input)
look_ahead_mask = create_look_ahead_mask(max_seq_len)
dec_padding_mask = create_padding_mask(enc_input)

outputs, attention_weights = model(enc_input, dec_input, enc_padding_mask, look_ahead_mask, dec_padding_mask)

print(f"Outputs: {outputs}")
print(f"Attention Weights: {attention_weights}")
