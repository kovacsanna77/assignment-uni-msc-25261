from simple_linear_regression import LinearRegressionGD


def run_demo() -> None:
    # Simple synthetic line: y = 3x + 2
    x = [0, 1, 2, 3, 4]
    y = [2, 5, 8, 11, 14]

    model = LinearRegressionGD(learning_rate=0.05, epochs=1200).fit(x, y)
    intercept, slope = model.parameters()

    prediction = model.predict([6])[0]
    mse = model.mean_squared_error(x, y)

    print(f"Learned intercept: {intercept:.3f}, slope: {slope:.3f}")
    print(f"Prediction for x=6: {prediction:.3f}")
    print(f"Training MSE: {mse:.6f}")


if __name__ == "__main__":
    run_demo()
