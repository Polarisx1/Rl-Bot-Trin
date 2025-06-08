"""Simple training script for the Rocket League neural network.

This script demonstrates how you might train the network using recorded
training data. Real training requires the RLBot framework to collect
examples of game states and desired actions.
"""

import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import RocketLeagueNet


class RecordedDataset(Dataset):
    """Dataset of recorded game states and actions."""

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
