import subprocess
import sys
from pathlib import Path

from simple_linear_regression import LinearRegressionGD


def test_end_to_end_training_and_prediction():
    x = [0, 1, 2, 3, 4, 5]
    y = [1.0, 3.1, 5.0, 6.9, 9.2, 10.8]

    model = LinearRegressionGD(learning_rate=0.05, epochs=1500).fit(x, y)
    mse = model.mean_squared_error(x, y)

    future_prediction = model.predict([6])[0]
    assert mse < 0.2
    assert 12.0 <= future_prediction <= 13.0


def test_demo_script_runs_via_subprocess():
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "src/demo_usage.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    output_lines = result.stdout.strip().splitlines()
    assert len(output_lines) == 3
