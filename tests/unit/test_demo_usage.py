import re

from demo_usage import run_demo


def test_run_demo_outputs_expected_lines(capsys):
    run_demo()
    captured = capsys.readouterr().out.splitlines()

    # Expect three lines with numeric content
    assert len(captured) == 3
    assert captured[0].startswith("Learned intercept:")
    assert captured[1].startswith("Prediction for x=6:")
    assert captured[2].startswith("Training MSE:")

    # Ensure numbers are present in the lines
    number_pattern = re.compile(r"-?\d+\.\d+")
    assert all(number_pattern.search(line) for line in captured)
