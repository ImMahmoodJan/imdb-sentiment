import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=128,
                 hidden_dim=256, num_layers=2,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits.squeeze(1)