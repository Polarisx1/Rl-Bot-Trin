import json
import os
import subprocess
import sys
from pathlib import Path
import importlib.util
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)


def test_training_creates_model(tmp_path):
    data = [{"features": [0.0]*20, "actions": [0.0]*6}]
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(data))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    result = subprocess.run(
        [sys.executable, "-m", "ml_bot.train", str(data_file), "--epochs", "1"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    model_path = Path(__file__).resolve().parents[1] / "ml_bot" / "model.pth"
    assert model_path.exists(), "model file not created"
    model_path.unlink()
