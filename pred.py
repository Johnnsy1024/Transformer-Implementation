import pickle as pkl

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./tokenizer.pkl", "rb") as fp:
    tokenizer = pkl.load(fp)

# 加载保存的模型
model = torch.load("./transformer.pth")
model.eval()


def translate(model, src, trg_max_length=128):
    # 推理按单个样本进行
    model.eval()
    src = src.to(device)

    bos_id = tokenizer.encode("[BOS]").ids[0]
    pad_id = tokenizer.encode("[PAD]").ids[0]
    eos_id = tokenizer.encode("[EOS]").ids[0]
    trg_raw = (
        torch.tensor(bos_id).unsqueeze(0).unsqueeze(0)
    )  # expand第二个维度为-1表示不动该维度的大小 [1, 1]

    for i in range(trg_max_length - 1):
        trg = F.pad(trg_raw, (0, trg_max_length - i - 1), "constant", pad_id)  # 填充长度
        output = model(src, trg)  # output: [1, seq_len, vocab_size]
        next_token = output[:, i + 1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        trg_raw = torch.cat([trg_raw, next_token], dim=-1)

        # 如果生成了结束标记（假设为1），则停止生成
        if next_token.item() == eos_id:
            break

    return trg


# 对测试集进行翻译
model.to(device)
translations = []
references = []

with torch.no_grad():
    for batch_idx, (src, trg) in enumerate(test_dataloader):
        translated = translate(model, src)
        translations.extend(translated.tolist())
        references.extend(trg.tolist())

        if batch_idx % 10 == 0:
            logger.info(f"已翻译 {batch_idx * test_dataloader.batch_size} 个样本")
