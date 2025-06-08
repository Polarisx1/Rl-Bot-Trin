# Rl-Bot-Trin

This repository contains a simple template for building machine learning
bots for Rocket League using the [RLBot](https://github.com/RLBot/RLBot)
framework and PyTorch.

## Getting Started

1. Install Python 3.8+ and Rocket League.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Gather training data in JSON format. Each sample should contain a
   `features` list (game state values) and an `actions` list (controller
   outputs). See `ml_bot/train.py` for details.
4. Train the model:

   ```bash
   python -m ml_bot.train path/to/training_data.json --epochs 20
   ```

5. Create `rlbot.cfg` to register your bot with RLBot:

   ```ini
   [HUMAN]
   team = 0

   [PYTHON_BOT_ML]
   python_file = ml_bot/bot.py
   class = MLBot
   team = 1
   ```

   See the [RLBot configuration docs](https://github.com/RLBot/RLBot/wiki/rlbot.cfg)
   for additional details. Launch the match using the RLBot GUI or the
   `rlbot-run` command.

## Project Structure

- `ml_bot/model.py` – PyTorch neural network definition.
- `ml_bot/train.py` – Example training script.
- `ml_bot/bot.py` – RLBot agent that loads the trained model.
- `requirements.txt` – Python dependencies.

## Reward System Guide

See [docs/reward_guide.md](docs/reward_guide.md) for tips on crafting reward functions.
The guide discusses competitive rewards (scoring, defending, boost usage) as well as
freestyle considerations like aerials and combos. It also explains how to balance positive
and negative rewards, use shaping effectively, and lists useful `GameTickPacket` metrics.
