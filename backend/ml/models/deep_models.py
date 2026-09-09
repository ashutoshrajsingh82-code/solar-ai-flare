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
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.proj = nn.Linear(cnn_channels, d_model)
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
        z = self.transformer(z)           # (batch, reduced_seq_len, d_model)
        z = z.permute(0, 2, 1)
        z = self.pool(z).squeeze(-1)      # (batch, d_model)
        return self.head(z)


DEEP_MODEL_BUILDERS = {
    "cnn_1d": lambda n_channels=2: CNN1D(n_channels=n_channels),
    "lstm": lambda n_channels=2: LSTMModel(n_channels=n_channels),
    "cnn_lstm_fusion": lambda n_channels=2: CNNLSTMHybrid(n_channels=n_channels),
    "cnn_transformer": lambda n_channels=2: CNNTransformerHybrid(n_channels=n_channels),
}
