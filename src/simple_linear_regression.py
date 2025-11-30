from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class LinearRegressionGD:
    """A tiny linear regression that learns with batch gradient descent."""

    learning_rate: float = 0.01
    epochs: int = 500
    _intercept: float = field(init=False, default=0.0)
    _slope: float = field(init=False, default=0.0)
    _fitted: bool = field(init=False, default=False)

    def fit(self, x: Sequence[float], y: Sequence[float]) -> "LinearRegressionGD":
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if not x:
            raise ValueError("training data cannot be empty")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("learning_rate and epochs must be positive")

        n = len(x)
        for _ in range(self.epochs):
            predictions = [self._predict_one(xi) for xi in x]
            error_intercept = sum(p - yi for p, yi in zip(predictions, y)) / n
            error_slope = sum((p - yi) * xi for p, yi, xi in zip(predictions, y, x)) / n
            self._intercept -= self.learning_rate * error_intercept
            self._slope -= self.learning_rate * error_slope

        self._fitted = True
        return self

    def predict(self, x: Iterable[float]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("call fit before predict")
        return [self._predict_one(xi) for xi in x]

    def mean_squared_error(self, x: Sequence[float], y: Sequence[float]) -> float:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        predictions = self.predict(x)
        errors = [(p - yi) ** 2 for p, yi in zip(predictions, y)]
        return sum(errors) / len(errors)

    def parameters(self) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("model has not been fitted")
        return self._intercept, self._slope

    def _predict_one(self, x_value: float) -> float:
        return self._intercept + self._slope * x_value
