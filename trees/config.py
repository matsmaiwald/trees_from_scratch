from dataclasses import dataclass
from typing import Tuple
from trees.model import OLSModel, AutoTreeStump, BoostedOLS, DecisionTree
from trees.model import ModelContainer


@dataclass
class DataConfig:
    N: int
    X_MIN: float
    X_MAX: float
    X_CHANGE_POINTS: Tuple[float]
    SLOPE_CHANGES: Tuple[float]
    SIGMA: float


data_config = DataConfig(
    N=100,
    X_MIN=0,
    X_MAX=10,
    X_CHANGE_POINTS=(0, 4, 6, 7),
    SLOPE_CHANGES=(3, -5, 5, -6),
    SIGMA=0.25,
)

models_decision_trees = {
    "decision_tree_1": ModelContainer(
        model=DecisionTree(min_leaf_size=35),
        plot_colour="red",
        plot_label="Min Leaf Size = 35",
    ),
    "decision_tree_2": ModelContainer(
        model=DecisionTree(min_leaf_size=20),
        plot_colour="red",
        plot_label="Min Leaf Size = 20",
    ),
    "decision_tree_3": ModelContainer(
        model=DecisionTree(min_leaf_size=5),
        plot_colour="red",
        plot_label="Min Leaf Size = 5",
    ),
}

models_boosted_trees = {
    "boosted_1": ModelContainer(
        model=BoostedOLS(n_boost=1, alpha=1),
        plot_colour="purple",
        plot_label=r"# boosting rounds = 1",
    ),
    "boosted_2": ModelContainer(
        model=BoostedOLS(n_boost=2, alpha=1),
        plot_colour="purple",
        plot_label=r"# boosting rounds = 2",
    ),
    "boosted_3": ModelContainer(
        model=BoostedOLS(n_boost=10, alpha=1),
        plot_colour="purple",
        plot_label=r"# boosting rounds = 10",
    ),
}
