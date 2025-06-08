"""Command line tool for generating new Rocket League bot packages."""

import argparse
from pathlib import Path
import sys

BOT_TEMPLATE = '''"""Example RLBot agent using a neural network model."""
import os
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameTickPacket

from .model import RocketLeagueNet


class MLBot(BaseAgent):
    def initialize_agent(self):
        """Load the ML model when the bot starts."""
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.model: RocketLeagueNet = RocketLeagueNet()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Compute controls from the game packet."""
        controller = SimpleControllerState()
        features = self._build_features(packet)
        with torch.no_grad():
            actions = self.model(features)
        controller.throttle = float(actions[0])
        controller.steer = float(actions[1])
        controller.pitch = float(actions[2])
        controller.yaw = float(actions[3])
        controller.roll = float(actions[4])
        controller.jump = actions[5] > 0
        return controller

    def _build_features(self, packet: GameTickPacket) -> torch.Tensor:
        """Convert game state to a tensor of features."""
        ball = packet.game_ball.physics.location
        car = packet.game_cars[self.index].physics.location
        features = [
            ball.x, ball.y, ball.z,
            car.x, car.y, car.z,
            packet.game_info.seconds_elapsed,
        ]
        features += [0.0] * (self.model.net[0].in_features - len(features))
        return torch.tensor(features, dtype=torch.float32)
'''

MODEL_TEMPLATE = '''import torch
import torch.nn as nn


class RocketLeagueNet(nn.Module):
    """Simple neural network for Rocket League bots."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''

TRAIN_TEMPLATE = '''"""Simple training script for the Rocket League neural network."""
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import RocketLeagueNet


class RecordedDataset(Dataset):
    def __init__(self, data_file: Path):
        with open(data_file, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        return features, actions


def train(data_path: str, epochs: int = 10, batch_size: int = 64):
    dataset = RecordedDataset(Path(data_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RocketLeagueNet(input_dim=len(dataset[0][0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for features, actions in loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.4f}")

    model_path = Path(__file__).with_name('model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Rocket League bot model")
    parser.add_argument("data", help="Path to JSON file containing training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    train(args.data, args.epochs)
'''

CFG_TEMPLATE = '''[HUMAN]
team = 0

[PYTHON_BOT_ML]
python_file = {name}/bot.py
class = MLBot
team = 1
'''


def create_new_bot(name: str, create_model: bool = False) -> None:
    path = Path(name)
    if path.exists():
        print(f"Directory '{name}' already exists", file=sys.stderr)
        sys.exit(1)
    path.mkdir(parents=True)

    (path / 'bot.py').write_text(BOT_TEMPLATE)
    (path / 'model.py').write_text(MODEL_TEMPLATE)
    (path / 'train.py').write_text(TRAIN_TEMPLATE)
    (path / 'rlbot.cfg').write_text(CFG_TEMPLATE.format(name=name))

    if create_model:
        (path / 'model.pth').touch()

    print(f"Created new bot package in {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bot-maker')
    sub = parser.add_subparsers(dest='command')

    new_parser = sub.add_parser('new', help='Create a new bot package')
    new_parser.add_argument('name', help='Name of the new bot module')
    new_parser.add_argument('--model', action='store_true', help='Create empty model.pth')

    args = parser.parse_args(argv)

    if args.command == 'new':
        create_new_bot(args.name, args.model)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
