import subprocess
import sys
from pathlib import Path
import os


def test_bot_maker_creates_files(tmp_path):
    temp_dir = tmp_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "bot_maker", "new", "temp_bot"],
        cwd=temp_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    bot_dir = temp_dir / "temp_bot"
    expected_files = ["bot.py", "rlbot.cfg"]
    for fname in expected_files:
        assert (bot_dir / fname).exists(), f"{fname} not created"

