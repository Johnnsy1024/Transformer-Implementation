from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from data_tokenize import TranslateDataset
from Decoder import Decoder
from EarlyStop import EarlyStopping
from Encoder import Encoder
from lr_scheduler import TransformerLRScheduler

# 生成输入的数据集
train_dataset = TranslateDataset(
    data_name=Path("./data/raw_data.csv"), dateset_type="train", use_cache=True
)
eval_dataset = TranslateDataset(
    data_name=Path("./data/raw_data.csv"), dateset_type="eval", use_cache=True
)
test_dataset = TranslateDataset(
    data_name=Path("./data/raw_data.csv"), dateset_type="test", use_cache=True
)


# 生成DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

vocab_size = train_dataset.vocab_size
tokenizer = train_dataset.tokenizer
src_max_length = train_dataset.src_max_length
trg_max_length = train_dataset.trg_max_length


# 构建完整的Transformer
class Transformer(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, vocab_size: int, device: str
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder_k_linear = nn.Linear(
            self.encoder.embed_size, self.encoder.embed_size, device=device
        )
        self.encoder_v_linear = nn.Linear(
            self.encoder.embed_size, self.encoder.embed_size, device=device
        )
        self.output_linear = nn.Linear(self.encoder.embed_size, vocab_size, device=device)
        self.num_heads = self.encoder.num_heads

    def forward(self, src: torch.tensor, trg: torch.tensor):
        src = src.to(device)
        trg = trg.to(device)
        batch_size, seq_len = src.shape[0], src.shape[1]
        encoder_res = self.encoder(
            src
        )  # encoder_res: [batch_size, seq_length, embedding_size]
        encoder_decoder_key = self.encoder_k_linear(encoder_res).reshape(
            (batch_size, seq_len, -1)
        )
        encoder_decoder_value = self.encoder_v_linear(encoder_res).reshape(
            (batch_size, seq_len, -1)
        )
        decoder_res = self.decoder(trg, encoder_decoder_key, encoder_decoder_value)

        output = self.output_linear(decoder_res)  # [batch_size, seq_len, vocab_size]

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(vocab_size)
decoder = Decoder(vocab_size)
model = Transformer(encoder, decoder, vocab_size=vocab_size, device=device)

EPOCH = 10
BATCH_SIZE = 128

early_stopping = EarlyStopping(patience=5, verbose=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, 512, 500)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    model.train()
    for batch_idx, (src, trg) in enumerate(train_dataloader):
        output = model(src, trg)
        loss = loss_func(
            output.view(
                -1,
                vocab_size,
            ).float(),
            trg.view(-1).to(device),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}, batch {batch_idx + 1}: loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}"
        )
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        test_loss = []
        for batch_idx, (src, trg) in enumerate(eval_dataloader):
            output = model(src, trg)
            loss = loss_func(
                output.view(
                    -1,
                    vocab_size,
                ).float(),
                trg.view(-1).to(device),
            )
            test_loss.append(loss.item())
    logger.info(f"Epoch {epoch + 1}, Eval loss: {sum(test_loss) / len(test_loss)}")
    early_stopping(sum(test_loss) / len(test_loss), model)
    if early_stopping.early_stop:
        print("Early stopping")
        break


torch.save(model, "./transformer.pth")


# 将索引转换回文本
# def indices_to_text(indices, vocab):
#     return " ".join([vocab.itos[idx] for idx in indices if idx not in [0, 1]])


# vocab = train_dataset.vocab  # 假设vocab在train_dataset中可用

# translated_texts = [indices_to_text(t, vocab) for t in translations]
# reference_texts = [indices_to_text(r, vocab) for r in references]

# # 打印一些翻译结果示例
# for i in range(5):
#     logger.info(f"源文本: {indices_to_text(test_dataset[i][0], vocab)}")
#     logger.info(f"参考翻译: {reference_texts[i]}")
#     logger.info(f"模型翻译: {translated_texts[i]}")
#     logger.info("---")

# 这里可以添加评估指标，如BLEU分数等
