"""Example RLBot agent using a neural network model."""
from typing import Optional
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
        """Convert game state to a tensor of features.

        This method should be adapted to include relevant game state data.
        """
        ball = packet.game_ball.physics.location
        car = packet.game_cars[self.index].physics.location
        features = [
            ball.x, ball.y, ball.z,
            car.x, car.y, car.z,
            packet.game_info.seconds_elapsed,
        ]
        # pad to the model's input dimension
        features += [0.0] * (self.model.net[0].in_features - len(features))
        return torch.tensor(features, dtype=torch.float32)
