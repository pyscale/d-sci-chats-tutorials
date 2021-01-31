from typing import Optional
import numpy as np
from scipy.optimize import minimize


def loss_funct(y_actual: np.ndarray, y_pred: np.ndarray, p: float = 2.0) -> float:
    """
    This will look for the median powered absolute differences

    :param y_actual:
    :param y_pred:
    :param p:
    :return:
    """

    assert y_actual.shape[0] == y_pred.shape[0], "The shape of the predictions must match the actuals"
    assert not p <= 0., "We must use a non-zero, non-negative power"

    diff = np.abs(y_actual - y_pred)

    if p == 1.:
        return np.median(diff)
    else:
        return np.median(diff ** p)


def predict_funct(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    prediction function

    :param data: your normalized data
    :param weights: your model weights
    :return: your predictions
    """

    assert data.shape[1] == weights.shape[0], "The dimensions must match"

    return data.dot(weights)


class TutorialModel:

    def __init__(self, power: float = 1.0):

        self.p = power
        # this will be set once the model is fitted
        self.coef_: Optional[np.ndarray] = None

    def fit(self, training_features: np.ndarray, training_labels: np.ndarray):
        """

        :param training_features:
        :param training_labels:
        :return:
        """

        p = self.p

        # this is a temporary function
        def tmp_loss(x: np.ndarray) -> float:

            y_pred = predict_funct(training_features, x)

            return loss_funct(training_labels, y_pred, p=p)

        res = minimize(tmp_loss, x0=np.random.random((training_features.shape[1],)))

        self.coef_ = res.get("x")

    def predict(self, testing_features):
        """

        :param testing_features:
        :return:
        """
        return predict_funct(testing_features, self.coef_)

    def score(self, testing_features, testing_labels):
        """

        :param testing_features:
        :param testing_labels:
        :return:
        """

        y_pred = predict_funct(testing_features, self.coef_)

        return loss_funct(testing_labels, y_pred, p=self.p)


if __name__ == "__main__":

    print("Creating random data")
    x_data = np.random.random((2000, 10))

    # essentially, we want the sum of the weights to be 1.
    print("Creating random weights")
    random_weights = np.random.random((10,))
    random_weights /= random_weights.sum()

    print("Getting the actual labels")
    y_labels = predict_funct(x_data, random_weights)

    # we will just use the default parameters
    model = TutorialModel()

    print("Splitting the data")
    x_train = x_data[:1000, :]
    x_test = x_data[1000:, :]

    y_train = y_labels[:1000]
    y_test = y_labels[1000:]

    print("Training model")
    model.fit(x_train, y_train)

    print(f"Model metric on test set is: {model.score(x_test, y_test)}\n")

    print(f"Actual weights: \n{random_weights}\n")
    print(f"Model weights: \n{model.coef_}\n")
