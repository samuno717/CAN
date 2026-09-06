import torch.nn as nn

class AELSTM(nn.Module):
    def __init__(self, num_features: int = 26, hidden_size: int = 64, sequence_length: int = 64, num_layers: int = 1):
        super(AELSTM, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # Encoding
        encoded_seq, (hidden_n, cell_n) = self.encoder_lstm(x)

        last_hidden = hidden_n[-1]
        # Bridge (preparing for the decoder)
        repeated_hidden = last_hidden.unsqueeze(1).repeat(1, x.size(1), 1)
        # Decoding
        decoded_seq, _ = self.decoder_lstm(repeated_hidden)
        # Reconstruction
        reconstruction = self.output_layer(decoded_seq)

        return reconstruction