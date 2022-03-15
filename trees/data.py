import numpy as np
from typing import Tuple, Dict, Any
from trees.config import DataConfig
from matplotlib import pyplot as plt


def _get_y(x, x_change_points, slope_changes, sigma) -> float:
    epsilon = np.random.normal(loc=0, scale=sigma, size=1)
    y_temp = 0
    for x_change_point, slope_delta in zip(x_change_points, slope_changes):
        if x > x_change_point:
            y_temp += slope_delta * (x - x_change_point)
    y = y_temp + epsilon
    return y


def _gen_data_point(
    x_min, x_max, x_change_points, slope_changes, sigma
) -> Tuple[float, float]:
    x = np.random.uniform(low=x_min, high=x_max, size=1)[0]
    y = _get_y(
        x=x,
        x_change_points=x_change_points,
        slope_changes=slope_changes,
        sigma=sigma,
    )[0]

    return x, y


def get_data(config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    data_x, data_y = [], []
    for i in range(config.N):
        x, y = _gen_data_point(
            config.X_MIN,
            config.X_MAX,
            config.X_CHANGE_POINTS,
            config.SLOPE_CHANGES,
            config.SIGMA,
        )
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)
