from __future__ import annotations

import argparse
from pathlib import Path

BOT_TEMPLATE = """from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

class Bot(BaseAgent):
    def get_output(self, packet) -> SimpleControllerState:
        return SimpleControllerState()
"""

CFG_TEMPLATE = """[PYTHON_BOT]
python_file = bot.py
class = Bot
"""

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bot-maker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create a new bot project")
    new_parser.add_argument("name")

    args = parser.parse_args(argv)

    if args.command == "new":
        target = Path(args.name)
        target.mkdir(parents=True, exist_ok=True)
        (target / "bot.py").write_text(BOT_TEMPLATE)
        (target / "rlbot.cfg").write_text(CFG_TEMPLATE)

if __name__ == "__main__":
    main()
