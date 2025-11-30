import pytest

from simple_linear_regression import LinearRegressionGD


def test_fit_raises_on_length_mismatch():
    model = LinearRegressionGD()
    with pytest.raises(ValueError):
        model.fit([1, 2, 3], [1, 2])


def test_predict_raises_before_fit():
    model = LinearRegressionGD()
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])


def test_mean_squared_error_raises_on_length_mismatch():
    model = LinearRegressionGD().fit([0, 1], [1, 3])
    with pytest.raises(ValueError):
        model.mean_squared_error([0], [1, 2])


def test_parameters_converge_on_simple_line():
    x = [0, 1, 2, 3]
    y = [1, 3, 5, 7]  # y = 2x + 1
    model = LinearRegressionGD(learning_rate=0.05, epochs=800).fit(x, y)
    intercept, slope = model.parameters()
    assert intercept == pytest.approx(1.0, abs=0.1)
    assert slope == pytest.approx(2.0, abs=0.1)
