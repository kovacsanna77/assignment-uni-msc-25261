import argparse

from simple_linear_regression import LinearRegressionGD


def run_demo(learning_rate: float = 0.05, epochs: int = 1200, predict_x: float = 6) -> dict:
    x = [0, 1, 2, 3, 4]
    y = [2, 5, 8, 11, 14]  # y = 3x + 2

    model = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs).fit(x, y)
    intercept, slope = model.parameters()

    prediction = model.predict([predict_x])[0]
    mse = model.mean_squared_error(x, y)

    print(f"Training points: {len(x)} | predict_x={predict_x}")
    print(f"Learned intercept: {intercept:.3f}, slope: {slope:.3f}")
    print(f"Prediction for x={predict_x}: {prediction:.3f}")
    print(f"Training MSE: {mse:.6f}")

    return {"intercept": intercept, "slope": slope, "prediction": prediction, "mse": mse}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny linear regression demo.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent step size")
    parser.add_argument("--epochs", type=int, default=1200, help="Number of training epochs")
    parser.add_argument("--predict-x", type=float, default=6, help="Value to predict")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(learning_rate=args.learning_rate, epochs=args.epochs, predict_x=args.predict_x)
