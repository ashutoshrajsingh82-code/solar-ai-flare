"""
PyTorch deep sequence models for dual-channel (SoLEXS + HEL1OS) X-ray time series.

All models are multi-task: a shared encoder feeds two heads:
  - Nowcast head:  4-class softmax  (Quiet / Pre-Flare / Flare / Decay)
  - Forecast head: 5 sigmoid outputs (P(flare<=1h), <=3h, <=6h, <=12h, <=24h)

Input shape convention: (batch, sequence_length, n_channels) -- channels-last,
converted internally to (batch, n_channels, sequence_length) for Conv1d.
"""
import torch
import torch.nn as nn
import math

N_NOWCAST_CLASSES = 4
N_FORECAST_HORIZONS = 5


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.nowcast_head = nn.Linear(64, N_NOWCAST_CLASSES)
        self.forecast_head = nn.Linear(64, N_FORECAST_HORIZONS)

    def forward(self, z):
        h = self.shared(z)
        nowcast_logits = self.nowcast_head(h)
        forecast_logits = self.forecast_head(h)  # BCEWithLogitsLoss applied outside (raw logits)
        return nowcast_logits, forecast_logits


class CNN1D(nn.Module):
    """1D CNN feature extractor + multi-task heads. Uses BOTH input channels jointly."""

    def __init__(self, n_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = MultiTaskHead(96, dropout)

    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        z = self.conv(x).squeeze(-1)
        return self.head(z)


class LSTMModel(nn.Module):
    """LSTM sequence model + multi-task heads."""

    def __init__(self, n_channels: int = 2, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(hidden_size, dropout)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])  # (batch, hidden_size)
        return self.head(last_hidden)


class CNNLSTMHybrid(nn.Module):
    """
    Primary candidate architecture:
    Two-channel sequence -> Conv1D feature extractor -> BatchNorm -> ReLU -> Pool
    -> LSTM -> shared latent representation -> dual task heads.
    """

    def __init__(self, n_channels: int = 2, cnn_channels: int = 48, hidden_size: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(hidden_size, dropout)

    def forward(self, x):
        # x: (batch, seq_len, channels) -> conv expects (batch, channels, seq_len)
        z = x.permute(0, 2, 1)
        z = self.conv(z)                 # (batch, cnn_channels, reduced_seq_len)
        z = z.permute(0, 2, 1)           # (batch, reduced_seq_len, cnn_channels) for LSTM
        out, (h_n, c_n) = self.lstm(z)
        latent = self.dropout(h_n[-1])
        return self.head(latent)


class CNNTransformerHybrid(nn.Module):
    """
    OPTIONAL model. CNN local feature extractor + Transformer encoder + dual heads.
    Kept lightweight (small d_model/heads) to stay CPU-trainable on the demo dataset.
    """

    def __init__(self, n_channels: int = 2, cnn_channels: int = 48, d_model: int = 48,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2,
                 max_sequence_length: int = 4096):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.proj = nn.Linear(cnn_channels, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = MultiTaskHead(d_model, dropout)

    def forward(self, x):
        z = x.permute(0, 2, 1)
        z = self.conv(z)                  # (batch, cnn_channels, reduced_seq_len)
        z = z.permute(0, 2, 1)            # (batch, reduced_seq_len, cnn_channels)
        z = self.proj(z)                  # (batch, reduced_seq_len, d_model)
        z = self.positional_encoding(z)   # position dimension is temporal axis (dim=1)
        z = self.transformer(z)           # (batch, reduced_seq_len, d_model)
        z = z.permute(0, 2, 1)
        z = self.pool(z).squeeze(-1)      # (batch, d_model)
        return self.head(z)


class SinusoidalPositionalEncoding(nn.Module):
    """Batch-first sinusoidal positions added along the temporal dimension."""

    def __init__(self, embedding_dim: int, max_length: int = 4096):
        super().__init__()
        if embedding_dim <= 0 or max_length <= 0:
            raise ValueError("embedding_dim and max_length must be positive.")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / embedding_dim))
        encoding = torch.zeros(max_length, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        if embedding_dim > 1:
            encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Positional encoding expects [batch, sequence, embedding] input.")
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds configured maximum {self.encoding.shape[1]}.")
        return x + self.encoding[:, :x.shape[1]].to(device=x.device, dtype=x.dtype)


class DecompositionLSTM(nn.Module):
    """LSTM over aligned STL trend, seasonal, and residual channels.

    Input has shape [batch, sequence, 3 * raw_channel_count]. Decomposition is
    precomputed per historical forecast-origin window outside this module.
    """

    def __init__(self, n_channels: int = 6, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3, min_history: int = 4):
        super().__init__()
        if n_channels <= 0 or hidden_size <= 0 or num_layers <= 0 or min_history <= 0:
            raise ValueError("DLSTM dimensions and minimum history must be positive.")
        self.n_channels = n_channels
        self.min_history = min_history
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(hidden_size, dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != self.n_channels:
            raise ValueError(f"DLSTM expects [batch, sequence, {self.n_channels}] input.")
        if x.shape[1] < self.min_history:
            raise ValueError(f"DLSTM requires at least {self.min_history} historical steps.")
        if not torch.isfinite(x).all():
            raise ValueError("DLSTM input contains missing or invalid decomposition values.")
        _, (hidden, _) = self.lstm(x)
        return self.head(self.dropout(hidden[-1]))


# `dropout` is now forwarded to every architecture. Previously the value chosen
# in the Training UI was accepted by the API but silently ignored here.
DEEP_MODEL_BUILDERS = {
    "cnn_1d": lambda n_channels=2, dropout=0.3: CNN1D(n_channels=n_channels, dropout=dropout),
    "lstm": lambda n_channels=2, dropout=0.3: LSTMModel(n_channels=n_channels, dropout=dropout),
    "cnn_lstm_fusion": lambda n_channels=2, dropout=0.3: CNNLSTMHybrid(n_channels=n_channels, dropout=dropout),
    "cnn_transformer": lambda n_channels=2, dropout=0.3, **kwargs: CNNTransformerHybrid(
        n_channels=n_channels, dropout=dropout, **kwargs
    ),
    "dlstm": lambda n_channels=6, dropout=0.3, **kwargs: DecompositionLSTM(
        n_channels=n_channels, dropout=dropout, **kwargs
    ),
}
