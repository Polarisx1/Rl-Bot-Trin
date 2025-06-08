import torch
import torch.nn as nn


class RocketLeagueNet(nn.Module):
    """Simple neural network for Rocket League bots.

    This template network expects an input vector of game state features and
    outputs controller actions.
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # throttle, steer, pitch, yaw, roll, jump
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return controller outputs in the range [-1, 1]."""
        return self.net(x)
