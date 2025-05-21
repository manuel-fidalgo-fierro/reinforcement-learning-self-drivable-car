# controller.py
import torch
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    """
    A linear controller: z (128) + LSTM-hidden (256)  -> 3 continuous actions
    Total params: 288*3 + 3 = 867
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(288, 3, bias=True)          # 288->3

        # optional: tiny Gaussian init (Ha & Schmidhuber 2018)
        nn.init.normal_(self.fc.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z, h):
        """
        z  :  (batch, 128)       – VAE latent at current step
        h  :  (batch, 256)      – MDN-RNN hidden state at current step
        returns  (batch, 3)     – action vector in -1 … +1
        """
        x = torch.cat([z, h], dim=-1)                   # shape (batch, 288)
        a = torch.tanh(self.fc(x))                      # keep within action range
        return a
