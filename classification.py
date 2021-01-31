import numpy as np


def loss_funct(y_actual: np.ndarray, y_pred: np.ndarray, p: float = 2.0) -> float:
    """
    This will look for the median powered absolute differences

    :param y_actual: These are the actual values of binary based off of seen data
    :param y_pred: These are the probabilities of the actual values
    :param p: This is a power for the difference
    :return:
    """

    assert y_actual.shape[0] == y_pred.shape[0], "The shape of the predictions must match the actuals"
    assert not p <= 0., "We must use a non-zero, non-negative power"

    diff = np.abs(y_actual - y_pred)

    if p == 1.:
        return np.median(diff)
    else:
        return np.median(diff ** p)