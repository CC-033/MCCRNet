import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

# ===============================================================================
# 
# ===============================================================================

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig()
        self.bertmodel = BertModel(bertconfig)

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, n_class, dropout=0.0):
        super(SubNet, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_type, in_size, hidden_size, out_size, num_layers=1, dropout=0.0, bidirectional=False, kernel_size=3):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == 'rnn':
            # Note: Using LSTM for 'rnn' type as it's common. Change to nn.RNN if needed.
            print(f"--- Creating an RNN Encoder: Using nn.LSTM with hidden_size={hidden_size} ---")
            self.rnn = nn.LSTM(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
            # This linear layer projects the hidden state of EACH time step.
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), out_size)

        elif self.encoder_type == 'gru':
            print(f"--- Creating a GRU Encoder: Using nn.GRU with hidden_size={hidden_size} ---")
            self.rnn = nn.GRU(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
            # This linear layer projects the hidden state of EACH time step.
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), out_size)

        elif self.encoder_type == 'tcn':
            # This implementation of TCN returns a sequence.
            print("--- Creating a TCN Encoder ---")
            from torch.nn.utils import weight_norm
            layers = []
            num_channels = [hidden_size] * num_layers
            for i in range(num_layers):
                dilation_size = 2 ** i
                in_ch = in_size if i == 0 else hidden_size
                out_ch = hidden_size
                layers += [
                    weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                          padding=(kernel_size - 1) * dilation_size,
                                          dilation=dilation_size)),
                    nn.ReLU()
                ]
            self.network = nn.Sequential(*layers)
            # Final projection to match out_size
            self.fc = nn.Linear(hidden_size, out_size)

        elif self.encoder_type == 'mlp':
            # MLP by nature does not process sequences, so it returns a 2D tensor.
            print("--- Creating an MLP Encoder (NOTE: Returns 2D tensor, incompatible with temporal fusion) ---")
            layers = []
            input_dim = in_size
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_size
            layers.append(nn.Linear(hidden_size, out_size))
            self.network = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def forward(self, x, lengths=None):
        if self.encoder_type in ['rnn', 'gru']:
            # ==================== KEY MODIFICATION HERE ====================
            # The goal is to return the full sequence of outputs, not just the last hidden state.
            if lengths is None:
                lengths = torch.full((x.size(0),), x.size(1), device=x.device)
            
            # Pack sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
            # Get the output sequence from RNN/GRU
            packed_output, _ = self.rnn(packed_input)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Apply linear layer to each time step
            # output shape: [Batch, Time, Hidden * Directions] -> [Batch, Time, Out_size]
            output = self.fc(output)
            
            # Return the 3D tensor
            return output
            # =============================================================

        elif self.encoder_type == 'tcn':
            # TCN expects (B, D_in, T)
            x = x.transpose(1, 2)
            out = self.network(x)
            # Return to (B, T, D_out)
            out = out.transpose(1, 2)
            return self.fc(out)

        elif self.encoder_type == 'mlp':
            # MLP path remains unchanged: it pools and returns a 2D tensor.
            # This is correct behavior for an MLP encoder.
            x = x.mean(dim=1)
            return self.network(x)


def get_encoder(encoder_type, in_size, hidden_size, out_size, num_layers=1, dropout=0.0, bidirectional=False, kernel_size=3):
    return Encoder(
        encoder_type=encoder_type,
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        kernel_size=kernel_size
    )