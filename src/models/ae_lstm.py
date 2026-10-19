import torch
import torch.nn as nn


class AELSTM(nn.Module):
    def __init__(
        self,
        num_features: int = 26,
        hidden_size: int = 64,
        latent_dim: int = 16,
        sequence_length: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Bottleneck
        self.fc_latent = nn.Linear(hidden_size, latent_dim)
        self.fc_unpack = nn.Linear(latent_dim, hidden_size)

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Reconstruction
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        # [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = x.shape

        # Encoding
        # hidden_n shape: [num_layers, batch_size, hidden_size]
        _, (hidden_n, _) = self.encoder_lstm(x)
        last_hidden = hidden_n[-1]  # [batch_size, hidden_size]

        # Bottleneck
        latent = self.fc_latent(last_hidden)  # [batch_size, latent_dim]

        # Vector preparation for decoder
        unpacked = self.fc_unpack(latent)  # [batch_size, hidden_size]

        # [batch_size, seq_len, hidden_size]
        repeated = unpacked.unsqueeze(1).expand(-1, seq_len, -1)

        # Decoding
        decoded_seq, _ = self.decoder_lstm(repeated)  # [batch_size, seq_len, hidden_size]

        # Reconstruction
        reconstruction = self.output_layer(decoded_seq)  # [batch_size, seq_len, num_features]

        if return_latent:
            return reconstruction, latent

        return reconstruction