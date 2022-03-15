import numpy as np
from typing import Callable, Tuple, List, Union
from trees.metrics import get_rmse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _split_data(
    x: np.ndarray, cutoff_x: float, y: np.ndarray  # = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split x and y into parts where x-value is leq than <cutoff_x>.
    """
    mask_lower = x <= cutoff_x
    x_lower, x_higher = x[mask_lower], x[~mask_lower]
    # if y is not None:
    y_lower, y_higher = y[mask_lower], y[~mask_lower]
    return [x_lower, x_higher], [y_lower, y_higher]
    # else:
    # return (x_lower, x_higher)


class BaseModel:
    training_error: float

    def _get_training_error(self, x, y):
        y_hat = self.get_preds(x)
        return get_rmse(actuals=y, preds=y_hat)

    def get_preds(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


@dataclass
class ModelContainer:
    model: BaseModel
    plot_colour: str
    plot_label: str


class OLSModel(BaseModel):
    def fit(self, x: np.ndarray, y: np.ndarray):
        if not len(x) >= 2:
            raise ValueError("Can't fit a LS model on single data point.")

        y_ = y.mean()
        x_ = x.mean()

        self.b_1 = np.sum((y - y_) * (x - x_)) / np.sum((x - x_) ** 2)
        self.b_0 = y_ - self.b_1 * x_
        self.fitted = True
        self.training_error = self._get_training_error(x, y)

    def get_coeffs(self):
        assert self.fitted, "Model not fitted yet, no coefficients available."
        return self.b_0, self.b_1

    def get_preds(self, x: Union[np.ndarray, float]) -> np.ndarray:
        assert self.fitted, "Model not fitted yet, no predictions available."
        return self.b_0 + self.b_1 * x


class TreeStump(BaseModel):
    x_knot: float
    sub_models: list

    def fit(self, x: np.ndarray, y: np.ndarray, x_knot: float = None) -> None:
        self.sub_models = []
        if x_knot is None:
            self.x_knot = x.mean()
        else:
            self.x_knot = x_knot
        data_x, data_y = _split_data(x=x, y=y, cutoff_x=self.x_knot)
        for i in range(len(data_x)):  # data_x has two entries only
            sub_model = OLSModel()
            sub_model.fit(x=data_x[i], y=data_y[i])
            self.sub_models.append(sub_model)
        self.fitted = True

        self.training_error = self._get_training_error(x, y)

    def get_preds(self, x: np.ndarray) -> np.ndarray:
        assert self.fitted, "Model not fitted yet, no predictions available."
        preds = []
        for i in range(len(x)):
            if x[i] <= self.x_knot:
                model = self.sub_models[0]
            else:
                model = self.sub_models[1]
            y_pred = model.get_preds(x=x[i])
            preds.append(y_pred)
        return np.asarray(preds)


def _find_best_split(x: np.ndarray, y: np.ndarray, steps: int = 20) -> float:
    x_grid = np.linspace(start=x.min(), stop=x.max(), num=steps)[
        1:-1
    ]  # exclude edges
    training_errors = []
    for x_i in x_grid:
        model = TreeStump()
        try:
            model.fit(x=np.array(x), y=np.array(y), x_knot=x_i)
            training_errors.append(model.training_error)
        except ValueError:
            training_errors.append(np.inf)
    best_split = x_grid[np.argmin(training_errors)]

    return best_split


class AutoTreeStump(BaseModel):
    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 20) -> None:
        x_split = _find_best_split(x=x, y=y, steps=20)
        self.model = TreeStump()
        self.model.fit(x=x, y=y, x_knot=x_split)
        self.fitted = True

        self.training_error = self._get_training_error(x, y)

    def get_preds(self, x: np.ndarray) -> np.ndarray:
        assert self.fitted, "Model not fitted yet, no predictions available."
        return self.model.get_preds(x=x)


class BoostedOLS(BaseModel):
    def __init__(self, n_boost: int = 10, alpha: float = 1):
        self._models: List[AutoTreeStump] = []
        self.n_boost = n_boost
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        y_rest = y.copy()
        for i in range(self.n_boost):
            model_temp = AutoTreeStump()
            model_temp.fit(x, y_rest)
            y_rest -= self.alpha * model_temp.get_preds(x)
            logger.info(f"Iteration {i}, mean training error: {y_rest.mean()}")
            logger.info(f"Iteration {i}, max training error: {y_rest.max()}")
            self._models.append(model_temp)
        self.fitted = True

        self.training_error = self._get_training_error(x, y)

    def get_preds(self, x: np.ndarray) -> np.ndarray:
        assert self.fitted, "Model not fitted yet, no predictions available."
        preds = np.zeros(shape=x.shape)
        for i in range(self.n_boost):
            preds += self.alpha * self._models[i].get_preds(x)
        return preds


class DecisionNode:
    """
    A Decision Node has two child nodes and a cutoff value
    which determines when each of the nodes is chosen.

    Child nodes are either nodes themselves or, if they are a leaf, a model.
    """

    def __init__(
        self,
        cutoff_value: float,
        below_branch: Union["DecisionNode", OLSModel],
        above_branch: Union["DecisionNode", OLSModel],
    ):
        self.cutoff_value = cutoff_value
        self.below_branch = below_branch
        self.above_branch = above_branch


def predict_recursive(
    x: float, node: Union[DecisionNode, OLSModel]
) -> Union[Callable, np.ndarray]:
    """Recursively predict y, given x, using the input node."""

    # Base case: we've reached a leaf, so get a prediction
    if isinstance(node, OLSModel):
        return node.get_preds(x)  # OLS models expect array inputs

    # If we're not on a leaf, step down to the next lower level in the tree.
    if x <= node.cutoff_value:
        return predict_recursive(x, node.below_branch)
    else:
        return predict_recursive(x, node.above_branch)


def build_decision_tree(
    x: np.ndarray, y: np.ndarray, min_leaf_size: int = 5
) -> Union[DecisionNode, OLSModel]:
    """
    Recursively build a decision tree w/ OLS models on its leafs.
    """
    # check if should split further
    x_split = _find_best_split(x=x, y=y)
    data_x, data_y = _split_data(x=x, y=y, cutoff_x=x_split)

    data_x_below = data_x[0]
    data_x_above = data_x[1]
    data_y_below = data_y[0]
    data_y_above = data_y[1]

    if len(data_x_below) <= min_leaf_size or len(data_x_above) <= min_leaf_size:
        logger.info(f"Building a leaf w/ {len(x)} data points.")
        leaf_model = OLSModel()
        leaf_model.fit(x, y)
        return leaf_model
    else:
        below_branch = build_decision_tree(
            data_x_below, data_y_below, min_leaf_size
        )
        above_branch = build_decision_tree(
            data_x_above, data_y_above, min_leaf_size
        )

        return DecisionNode(
            cutoff_value=x_split,
            below_branch=below_branch,
            above_branch=above_branch,
        )


class DecisionTree(BaseModel):
    def __init__(self, min_leaf_size):
        self.min_leaf_size = min_leaf_size
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = build_decision_tree(
            x=x.copy(), y=y.copy(), min_leaf_size=self.min_leaf_size
        )
        self.training_error = self._get_training_error(x, y)

    def get_preds(self, x: np.ndarray) -> np.ndarray:
        _preds = []
        for i in range(len(x)):
            _preds.append(predict_recursive(x[i], self._model))
        return np.asarray(_preds)
