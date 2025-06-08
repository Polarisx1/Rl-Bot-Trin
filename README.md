# Rl-Bot-Trin

This repository contains a simple template for building machine learning
bots for Rocket League using the [RLBot](https://github.com/RLBot/RLBot)
framework and PyTorch.

## Getting Started

The following steps walk through training and running the baseline ML bot from
scratch.

1. **Install prerequisites**
   - Python 3.8 or newer.
   - Rocket League (the game must be installed to run RLBot).

2. **Install Python dependencies**

   All required packages are listed in `requirements.txt` and include
   `rlbot`, `numpy` and `torch`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Gather training data**

   Create a JSON file containing recorded game states and the corresponding
   controller actions. Each entry must contain a `features` list and an
   `actions` list (see `ml_bot/train.py` for the expected shapes).  An example
   looks like:

   ```json
   [
     {"features": [0.0, 1.2, ...], "actions": [0.5, -1.0, ...]},
     ...
   ]
   ```
4. **Train the model**

   ```bash
   # Run this command from the repository root so the ml_bot package resolves
   # correctly.
   python -m ml_bot.train path/to/training_data.json --epochs 20
   ```

5. **Create `rlbot.cfg`**
   Register the trained bot with RLBot:

   ```ini
   [HUMAN]
   team = 0

   [PYTHON_BOT_ML]
   python_file = ml_bot/bot.py
   class = MLBot
   team = 1
   ```

   See the [RLBot configuration docs](https://github.com/RLBot/RLBot/wiki/rlbot.cfg)
   for additional details.

6. **Run a match**

   Launch a game using the RLBot GUI or the `rlbot-run` command line tool. The
   bot will load the `model.pth` file produced by the training step and begin
   playing using the neural network's outputs.

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
