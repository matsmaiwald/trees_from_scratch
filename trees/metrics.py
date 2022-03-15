import numpy as np


def get_rmse(actuals: np.ndarray, preds: np.ndarray) -> float:
    assert actuals.shape == preds.shape
    errors = actuals - preds
    rmse = (np.mean(errors**2)) ** (1 / 2)
    return rmse
