# Minimal Transformer block (PyTorch) — starter
import torch
import torch.nn as nn

class MiniSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.ln(x + y)

class MiniFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(x + self.net(x))

class MiniBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.sa = MiniSelfAttention(d_model, n_heads)
        self.ffn = MiniFFN(d_model, d_ff)
    def forward(self, x):
        return self.ffn(self.sa(x))

if __name__ == "__main__":
    x = torch.randn(2, 16, 128)  # (batch, seq, d_model)
    block = MiniBlock()
    y = block(x)
    print("OK", y.shape)
