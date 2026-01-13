
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# --- Residual Convolutional Block ---
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.shortcut(x))

# --- Advanced CNN+LSTM Model with Residual Connections and Multi-Head Attention ---
class AdvancedCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        
        # Deeper CNN with Residual Connections
        # Note: Added ceil_mode=True to MaxPool1d to support single-frame inputs (sequence length 1)
        self.cnn = nn.Sequential(
            ResidualConvBlock(input_dim, 256),
            nn.MaxPool1d(2, ceil_mode=True),
            ResidualConvBlock(256, 128),
            nn.MaxPool1d(2, ceil_mode=True),
            ResidualConvBlock(128, 64),
            nn.MaxPool1d(2, ceil_mode=True),
            ResidualConvBlock(64, 64)
        )
        
        # Bi-LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, 
                            num_layers=num_layers, bidirectional=True, dropout=0.3)
        
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def get_feature(self, x, lengths):
        """
        Extracts the feature vector for the input sequence/frame.
        Returns the output of the global average pooling over the attention output.
        Shape: (Batch, Hidden*2)
        """
        # x shape: (B, T, F) where B=batch, T=time, F=features
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.cnn(x)  # (B, 64, T//8)
        x = x.permute(0, 2, 1)  # (B, T//8, 64)

        # Adjust lengths for pooling (divide by 8 due to three MaxPool1d layers)
        # Using ceil_mode=True logic: length is essentially divided by 8 but taking ceil at each step?
        # Simpler heuristic: length 1 -> 1.
        adj_len = (lengths.cpu() // 8).clamp(min=1)
        
        # Ensure x length and adj_len match for packing if needed, but for single frame T=1 (after pool T=1)
        # If lengths are original lengths, we need careful handling.
        # For single frame usage in run_landmark_model, T=1. Pool -> 1. adj_len=1.
        
        packed = pack_padded_sequence(x, adj_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out: (B, T_pooled, hidden*2)

        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global Average Pooling over the temporal dimension
        out = torch.mean(attn_out, dim=1)  # (B, hidden*2)
        return out

    def forward(self, x, lengths):
        out = self.get_feature(x, lengths)
        return self.fc(out)
