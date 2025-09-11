"""
We should outline the code first before actually coding.

This file implements a linear regression in numpy.
Loss function is MSE,
Optimization is done via good old gradient descent.

We start with univariate linear regression.

- We generate a fake data.
1. We sample the slope and intercept
2. sample x data from normal distribution
3. get y from x using ground truth parameters
4. add gaussian noise

- Then we need to initialize the model.
A univariate linear regression model consists of two parameters. Both initialized to 0.

- For gradient descent, we need the gradient. We implement the gradient function which will output
the gradient vector of size 2 each dimension representing gradient of slope and intercept for the MSE
loss function respectively.

- A step function does a single update pass. It will sample data from the training dataset, calculate
the gradient, and update the parameters a single time.

- `train` function calls above `step` function iteratively until either max iteration or threshold
of error is reached.

ETC:
- We use float64 since numpy random usually generates float64 data.

TODO:
- [ ] exit when error threshold is reached
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


@dataclass
class Model:
    slope: float = 0
    intercept: float = 0


def generate_fake_data(
    num_data: int,
    gt_loc: float = 0,
    gt_scale: float = 1,
    noise_loc: float = 0,
    noise_scale: float = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    1. We sample the slope and intercept
    2. sample x data from normal distribution
    3. get y from x using ground truth parameters
    4. add gaussian noise
    """
    if gt_scale <= 0 or noise_scale <= 0:
        raise ValueError(
            f"scale cannot be leq 0. Got: GT scale:{gt_scale} and Noise scale:{noise_scale}."
        )
    gt_parameters = np.random.normal(loc=gt_loc, scale=gt_scale, size=2)
    x_points = np.random.normal(size=num_data)
    gt_y_points = gt_parameters[0] * x_points + gt_parameters[1]
    assert isinstance(gt_y_points, np.ndarray)
    y_points = gt_y_points + np.random.normal(
        loc=noise_loc, scale=noise_scale, size=gt_y_points.shape
    )

    return x_points, y_points


def get_gradient(
    model: Model, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    - For gradient descent, we need the gradient. We implement the gradient function which will output
    the gradient vector of size 2 each dimension representing gradient of slope and intercept for the MSE
    loss function respectively.

    - MSE(x,y) = 1/N sum_(i=1)^N (y_i-(mx_i+b))^2
    - A(u) = u^2 = (y_i-(mx_i+b))^2
    - dA(u)/du = 2u = 2(y_i-(mx_i+b))
    - dB(m,b)/dm = -x
    - dB(m,b)/db = -1
    """
    u = y - (model.slope * x + model.intercept)
    return np.array((-2 * x * u, -1 * u))


def step(
    model: Model,
    x_points: npt.NDArray[np.float64],
    y_points: npt.NDArray[np.float64],
    lr: float = 0.001,
) -> Model:
    """
    - A step function does a single update pass. It will sample data from the training dataset, calculate
    the gradient, and update the parameters a single time.
    """
    gradients = get_gradient(model, x_points, y_points).mean(axis=-1)
    # negative because gradient points to the upward (increasing the loss function). But we want the opposite.
    return Model(
        slope=model.slope - lr * gradients[0],
        intercept=model.intercept - lr * gradients[1],
    )


def train(
    model: Model,
    x_points: npt.NDArray[np.float64],
    y_points: npt.NDArray[np.float64],
    lr: float = 0.001,
    max_iter: int = 1000,
) -> Model:
    """- `train` function calls above `step` function iteratively until either max iteration or threshold
    of error is reached."""
    for i in range(max_iter):
        model = step(model, x_points, y_points, lr)
    return model


def predict(
    model: Model,
    x_points: npt.NDArray[np.float64],
    y_points: npt.NDArray[np.float64],
) -> float:
    predictions = model.slope * x_points + model.intercept
    mse = np.pow(y_points - predictions, 2).mean()
    return mse


if __name__ == "__main__":
    num_data = 10000
    lr = 1e-3
    x_points, y_points = generate_fake_data(num_data)

    model = Model()
    model = train(model, x_points, y_points, lr)
    mse = predict(model, x_points, y_points)
    print(f"MSE: {mse:.3f}")
    print(f"Slope: {model.slope:.3f}")
    print(f"Intercept: {model.intercept:.3f}")

    xs = np.array([min(x_points), max(x_points)])
    ys = model.slope * xs + model.intercept

    plt.style.use("dark_background")
    plt.scatter(x_points, y_points)
    plt.plot(xs, ys)
    plt.show()
