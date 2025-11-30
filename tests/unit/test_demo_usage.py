import re

from demo_usage import run_demo


def test_run_demo_outputs_expected_lines(capsys):
    result = run_demo()
    captured = capsys.readouterr().out.splitlines()

    # Expect four lines with numeric content
    assert len(captured) == 4
    assert captured[0].startswith("Training points:")
    assert captured[1].startswith("Learned intercept:")
    assert captured[2].startswith("Prediction for x=")
    assert captured[3].startswith("Training MSE:")

    # Ensure numbers are present in the lines
    number_pattern = re.compile(r"-?\d+\.\d+")
    assert all(number_pattern.search(line) for line in captured)
    assert set(result.keys()) == {"intercept", "slope", "prediction", "mse"}
