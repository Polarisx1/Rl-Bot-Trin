# Bot Maker Guide

The RLBot package provides a `bot-maker` command-line tool which can generate a new bot project for you. Installing RLBot will place the script in your PATH along with the standard RLBot modules.

## Installation

If you don't already have RLBot installed, grab it from PyPI:

```bash
pip install rlbot
```

This installs the `bot-maker` script and all required dependencies.

## Creating a New Bot

Run `bot-maker new` followed by the name of your bot:

```bash
bot-maker new MyAwesomeBot
```

The command will create a new folder named `MyAwesomeBot` with example configuration files and a starter Python bot. You can then open the folder and begin customizing the code.

## Using the Generated Project

Within the new folder you'll find:

- `bot.py` – a minimal bot implementation.
- `rlbot.cfg` – configuration to register the bot with RLBot.
- Example loadout and team settings.

Open `bot.py` to start editing your bot's behavior. Launch a match using the RLBot GUI or the `rlbot-run` command to test changes immediately.

Pass `--help` to `bot-maker` for a list of additional options, including template choices and folder paths.

See the RLBot documentation for more details on the generated files and other `bot-maker` features.
