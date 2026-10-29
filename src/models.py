"""Teachers (large) and students (edge-sized) for per-flow classification.

Every model maps a (batch, n_features) float tensor to (batch, n_classes)
logits. Students expose ``features(x)`` (the penultimate activation) so
feature-level distillation can be added without changing their forward pass.
"""
from __future__ import annotations

import torch
from torch import nn


# ------------------------------------------------------------------ teachers
class CNNBiLSTMMHA(nn.Module):
    """Architecture of the CNN-BiLSTM-MHA paper (same layer sizes), adapted from
    10-flow windows to a single flow: the feature vector is read as a 1-D
    signal, the BiLSTM and attention run over the CNN's output positions."""

    def __init__(self, n_features, n_classes, channels=(64, 128, 256), lstm_hidden=256,
                 lstm_layers=2, heads=8, fc=(256, 128), dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.MaxPool1d(2, ceil_mode=True)]
            c_in = c
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(c_in, lstm_hidden, lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        d = 2 * lstm_hidden
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        head, h_in = [], d
        for h in fc:
            head += [nn.Linear(h_in, h), nn.ReLU(), nn.Dropout(0.4)]
            h_in = h
        self.head = nn.Sequential(*head, nn.Linear(h_in, n_classes))

    def forward(self, x):
        z = self.cnn(x.unsqueeze(1)).transpose(1, 2)        # (B, L, C)
        z, _ = self.lstm(z)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm(z + a).mean(dim=1)
        return self.head(z)


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Gorishniy et al., 2021): one token per
    feature plus a [CLS] token."""

    def __init__(self, n_features, n_classes, d_token=128, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        block = nn.TransformerEncoderLayer(d_token, heads, 2 * d_token, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.weight + self.bias               # (B, F, D)
        # x.size(0), not len(x): ONNX tracing would freeze len(x) as a constant.
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        return self.head(self.norm(self.encoder(tok)[:, 0]))


# ------------------------------------------------------------------ students
class MLPStudent(nn.Module):
    def __init__(self, n_features, n_classes, hidden=(64, 32)):
        super().__init__()
        layers, h_in = [], n_features
        for h in hidden:
            layers += [nn.Linear(h_in, h), nn.ReLU()]
            h_in = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(h_in, n_classes)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


class TinyCNNStudent(nn.Module):
    """Depthwise-separable 1-D CNN; flattened (not pooled) so the head keeps
    feature positions, which carry meaning in tabular data."""

    def __init__(self, n_features, n_classes, channels=(8, 16, 32)):
        super().__init__()
        layers, c_in = [nn.Conv1d(1, channels[0], 5, stride=2, padding=2), nn.ReLU()], channels[0]
        for c in channels[1:]:
            layers += [nn.Conv1d(c_in, c_in, 3, stride=2, padding=1, groups=c_in),
                       nn.Conv1d(c_in, c, 1), nn.ReLU()]
            c_in = c
        self.body = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.body(torch.zeros(1, 1, n_features)).shape[1]
        self.head = nn.Linear(n_flat, n_classes)

    def features(self, x):
        return self.body(x.unsqueeze(1))

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {
    "cnn_bilstm_mha": CNNBiLSTMMHA,
    "ft_transformer": FTTransformer,
    "mlp_s": MLPStudent,
    "cnn_s": TinyCNNStudent,
}
TEACHERS = ("cnn_bilstm_mha", "ft_transformer")
STUDENTS = ("mlp_s", "cnn_s")

# Student-size sweep (experiment E5): MLP with hidden (h, h/2).
# mlp_h64 has the same architecture as mlp_s.
SWEEP_WIDTHS = (8, 16, 32, 64, 128, 256, 512)
for _h in SWEEP_WIDTHS:
    MODELS[f"mlp_h{_h}"] = (lambda h: lambda f, c: MLPStudent(f, c, hidden=(h, max(h // 2, 4))))(_h)
SWEEP = tuple(f"mlp_h{h}" for h in SWEEP_WIDTHS)


def build(name: str, n_features: int, n_classes: int) -> nn.Module:
    return MODELS[name](n_features, n_classes)


def size_report(model: nn.Module) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {"params": n, "size_kb_fp32": round(n * 4 / 1024, 1), "size_kb_int8": round(n / 1024, 1)}
