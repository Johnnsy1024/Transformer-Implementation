import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_x: torch.tensor):
        batch_size, seq_len, embed_size = input_x.shape
        pos_encoding = torch.zeros(batch_size, seq_len, embed_size)
        pos_encoding_element = torch.arange(0, seq_len)[..., None].tile(
            (1, embed_size)
        ) / (
            10000
            ** ((torch.arange(embed_size) // 2) / embed_size)[None, ...].tile(seq_len, 1)
        )
        pos_encoding[..., 0::2] = torch.sin(pos_encoding_element[..., 0::2])
        pos_encoding[..., 1::2] = torch.cos(pos_encoding_element[..., 1::2])
        return pos_encoding


class InputBlock(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_size)

    def _positional_encoding(self, input_x: torch.tensor):
        batch_size, seq_len, embed_size = input_x.shape
        pos_encoding = torch.zeros(batch_size, seq_len, embed_size)
        pos_encoding_element = torch.arange(0, seq_len)[..., None].tile(
            (1, embed_size)
        ) / (
            10000
            ** ((torch.arange(embed_size) // 2) / embed_size)[None, ...].tile(seq_len, 1)
        )
        pos_encoding[..., 0::2] = torch.sin(pos_encoding_element[..., 0::2])
        pos_encoding[..., 1::2] = torch.cos(pos_encoding_element[..., 1::2])
        return pos_encoding

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len]
        x = self.input_embedding(input_x.long())  # x: [batch_size, seq_len, embed_size]
        pos_embedding = self._positional_encoding(x).to(input_x.device)
        x = x + pos_embedding

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        src_mask_flag: bool = True,
    ):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding维度必须可被num_heads整除"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.src_mask_flag = src_mask_flag
        self.linear_key = nn.Linear(embed_size, embed_size)
        self.linear_query = nn.Linear(embed_size, embed_size)
        self.linear_value = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def _gen_src_mask(self, src, src_padding_idx: int = 0):
        # src: [batch_size, seq_len]
        src_mask = (
            torch.where((src != src_padding_idx), 0, -1e20).unsqueeze(1).unsqueeze(1)
        )  # [batch_size, 1, 1, seq_len]如果只保持[batch_size, seq_len]的形状的话，
        # 无法与[batch_size, num_heads, seq_len, seq_len]形状的attention进行广播
        return src_mask

    def forward(self, input_x: torch.tensor, x_raw: torch.tensor):
        # input_x: [batch_size, seq_len, embed_size]
        batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        key = self.linear_key(input_x).reshape((batch_size, seq_len, self.num_heads, -1))
        query = self.linear_query(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        value = self.linear_value(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        attention = (
            key.permute(0, 2, 1, 3) @ query.permute(0, 2, 3, 1) / self.embed_size**0.5
        )  # [batch_size, num_heads, seq_len, seq_len]
        if self.src_mask_flag:
            src_mask = self._gen_src_mask(x_raw)
            attention = attention + src_mask
        out = (F.softmax(attention, dim=-1) @ value.permute(0, 2, 1, 3)).reshape(
            (batch_size, seq_len, -1)
        )  # out: [batch_size, seq_len, embed_size]
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(
        self, embed_size: int = 512, ffn_hidden_size: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.linear_1 = nn.Linear(embed_size, ffn_hidden_size)
        self.linear_2 = nn.Linear(ffn_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len ,embed_size]
        x = self.linear_1(input_x)
        x = F.relu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        src_mask_flag: bool = True,
    ):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding维度必须可被num_heads整除"
        self.multi_head_attention = MultiHeadAttention(
            embed_size, num_heads, dropout, src_mask_flag
        )
        self.ffn = FeedForward(embed_size, ffn_hidden_size, dropout)
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(
        self, input_x: torch.tensor, x_raw: torch.tensor
    ):  # input_x: [batch_size, seq_len, embed_size]
        out_1 = self.layernorm(self.multi_head_attention(input_x, x_raw) + input_x)
        out_2 = self.layernorm(self.ffn(out_1) + out_1)
        return out_2


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 64,
        src_mask_flag: bool = True,
        block_num: int = 6,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.src_mask_flag = src_mask_flag
        self.num_heads = num_heads
        self.dropput = dropout
        self.fnn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.block_num = block_num
        self.dropout = dropout
        self.ffn_hidden_size = ffn_hidden_size

        self.input_block = InputBlock(vocab_size, embed_size)

        self.encoder_block = EncoderBlock(
            self.embed_size,
            self.num_heads,
            self.dropout,
            self.ffn_hidden_size,
            src_mask_flag,
        )
        self.encoder = nn.ModuleList([self.encoder_block for _ in range(self.block_num)])

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len]
        out = self.input_block(input_x)
        for layer in self.encoder:
            out = layer(out, input_x)
        # x = self.encoder(out_1, input_x)
        return out


if __name__ == "__main__":

    device = torch.device("mps")
    input_x = torch.randint(low=0, high=32, size=(32, 128), device=device)
    model = Encoder(vocab_size=30000, src_mask_flag=True, device=device)
    model.to(device)
    res = model(input_x)
