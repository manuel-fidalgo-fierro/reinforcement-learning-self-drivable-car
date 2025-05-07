import torch
import torch.nn as nn
import torch.nn.functional as F
from models.device import get_device_name

class MDNRNN(nn.Module):
    """
    Mixture Density Network - Recurrent Neural Network (MDN-RNN)
    
    This model combines an LSTM with a Mixture Density Network to predict the next state
    in a continuous action space. It takes as input:
    - z: latent vector from VAE (128 dimensions)
    - action: current action (2 dimensions)
    - reward: previous reward (1 dimension)
    
    The total input dimension is 132 (128 + 3 + 1).
    The LSTM processes this into a 256-dimensional hidden state,
    which is then transformed into 1920 parameters for the mixture distribution.
    """
    
    def __init__(self, z_dim=128, action_dim=2, reward_dim=1, hidden_dim=256, num_mixtures=5, dropout=0.1):
        super(MDNRNN, self).__init__()
        
        # Store dimensions
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        
        # Input dimension is sum of z, action, and reward dimensions
        self.input_dim = z_dim + action_dim + reward_dim
        
        # LSTM layer
        # TODO consider using transformer instead of LSTM.
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        
        # MDN layer
        # For each mixture component we need:
        # - z_dim means (128)
        # - z_dim variances (128)
        # - 1 mixture weight
        # Total parameters per mixture: 128 + 128 + 1 = 257
        # Total parameters for all mixtures: 257 * num_mixtures = 1285
        self.mdn = nn.Linear(hidden_dim, 257 * num_mixtures)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the MDN-RNN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
               where input_dim = z_dim + action_dim + reward_dim
            hidden: Initial hidden state for LSTM (optional)
            
        Returns:
            mdn_params: Parameters for the mixture distribution
            hidden: Final hidden state of the LSTM
        """
        # Process through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Transform LSTM output to MDN parameters
        mdn_params = self.mdn(lstm_out)
        
        return mdn_params, hidden
    
    def get_mixture_params(self, mdn_params):
        """
        Extract the mixture parameters from the MDN output.
        
        Args:
            mdn_params: Output from the MDN layer
            
        Returns:
            pi: Mixture weights
            mu: Means for each mixture component
            sigma: Standard deviations for each mixture component
        """
        # Reshape the parameters
        batch_size = mdn_params.size(0)
        seq_len = mdn_params.size(1)
        
        # Split the parameters into pi, mu, and sigma
        params = mdn_params.view(batch_size, seq_len, self.num_mixtures, -1)
        
        # Extract parameters
        pi = F.softmax(params[..., 0], dim=-1)  # Mixture weights
        mu = params[..., 1:self.z_dim+1]  # Means
        sigma = torch.exp(params[..., self.z_dim+1:])  # Standard deviations
        
        return pi, mu, sigma
    
    @staticmethod
    def load_mdn_rnn_model(model_path='models/mdn_rnn/mdn_rnn_final.pt'):
        """Load the trained MDN-RNN model."""
        device = get_device_name()
        print(f"Using device: {device}")
        
        # Initialize model
        model = MDNRNN().to(device)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, device