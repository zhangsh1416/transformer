import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import Transformer
from utils import create_padding_mask, create_look_ahead_mask
from tqdm import tqdm
from torchsummary import summary
import logging

# 设置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 装饰器，用于日志记录
def log_decorator(func):
    def wrapper(*args, **kwargs):
        logging.info(f'Running {func.__name__}...')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__} finished.')
        return result
    return wrapper

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
num_epochs = 10

# 数据生成（假设我们有输入和目标张量enc_input, dec_input, 和dec_target）
enc_input = torch.randint(0, input_vocab_size, (batch_size, max_seq_len))
dec_input = torch.randint(0, target_vocab_size, (batch_size, max_seq_len))
dec_target = torch.randint(0, target_vocab_size, (batch_size, max_seq_len))

dataset = TensorDataset(enc_input, dec_input, dec_target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型初始化
model = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 模型摘要
summary(model, [(max_seq_len,), (max_seq_len,)])

@log_decorator
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        enc_input, dec_input, dec_target = batch

        enc_padding_mask = create_padding_mask(enc_input)
        look_ahead_mask = create_look_ahead_mask(max_seq_len)
        dec_padding_mask = create_padding_mask(enc_input)

        optimizer.zero_grad()

        outputs, _ = model(enc_input, dec_input, enc_padding_mask, look_ahead_mask, dec_padding_mask)
        loss = criterion(outputs.view(-1, target_vocab_size), dec_target.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@log_decorator
def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    train(model, dataloader, criterion, optimizer, num_epochs)
