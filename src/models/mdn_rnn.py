import torch
import torch.nn as nn

class MDNRNN(nn.Module):
    """
    MDN-RNN model:

    input:  (batch_size, seq_len, input_size)
    LSTM:   hidden_size=256, returns
            - output_seq: (batch_size, seq_len, 256)
            - (h_n, c_n): each (1, batch_size, 256)
    Dense:  maps each of the 256 outputs to 1921 MDN params:
            output_seq: (batch_size, seq_len, 1921)
    """
    def __init__(self, input_size=131, hidden_state_size=256, num_mixtures=5, z_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_state_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Time‐distributed dense layer to produce MDN parameters
        self.fc = nn.Linear(hidden_state_size, num_mixtures * 3 * z_dim) # 3 because of pi, mu, log_sigma

    def forward(self, x, hidden=None):
        """
        x:      Tensor of shape (batch_size, seq_len, input_size)
        hidden: optional tuple (h_0, c_0), each of shape (num_layers, batch_size, hidden_size)

        returns:
          mdn_params: Tensor of shape (batch_size, seq_len, output_size)
          hidden:     tuple (h_n, c_n) of the final LSTM states
        """
        # run through LSTM
        # out: (batch_size, seq_len, hidden_size)
        # h_n, c_n: each (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x, hidden)

        # project each time step to MDN parameters
        # mdn_params: (batch_size, seq_len, output_size)
        mdn_params = self.fc(out)
        return mdn_params, (h_n, c_n)
