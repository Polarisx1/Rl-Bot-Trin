# Bot Maker Guide

The RLBot package provides a `bot-maker` command-line tool which can generate
a new bot project for you. Installing RLBot will place the script in your PATH
along with the standard RLBot modules.

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

The command will create a new folder named `MyAwesomeBot` with example
configuration files and a starter Python bot. You can then open the folder and
begin customizing the code.

See the RLBot documentation for more details on the generated files and
additional `bot-maker` options.
