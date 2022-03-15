from typing import Dict
from trees.model import ModelContainer
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

FIG_SIZE_X = 8
FIG_SIZE_Y = 8
COLORS = color_palette("colorblind", 4)


def train_models(
    models: Dict[str, ModelContainer], x: np.ndarray, y: np.ndarray
):
    for model_name, model_container in models.items():
        model_container.model.fit(x=x, y=y)
        print(f"Model name: {model_name}")
        print(
            f"Training error: {round(model_container.model.training_error, 3)}"
        )
        print(f"---" * 10)


def plot_actuals(x: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y / 3))
    ax.scatter(x, y, marker=".", label="Actuals")
    ax.legend(loc="upper right")
    return fig


def plot_models_vs_actuals(
    models: Dict[str, ModelContainer], x: np.ndarray, y: np.ndarray
):
    fig, axs = plt.subplots(
        figsize=(FIG_SIZE_X, FIG_SIZE_Y), nrows=3, sharex=True
    )
    marker_index = 0
    markers = ["o", "v", "^", "s"]
    for model_name, model_container in models.items():
        axs[marker_index].scatter(
            x,
            y,
            marker=".",
            label="Actuals",
            # s=150,
            color=COLORS[0],
        )
        axs[marker_index].scatter(
            x,
            model_container.model.get_preds(x),
            marker=markers[marker_index],
            # s=80,
            alpha=1,
            color=COLORS[marker_index + 1],
            # color=model_container.plot_colour,
            label=model_container.plot_label,
        )
        # axs[marker_index].legend()
        marker_index += 1
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]  # concatenate
    by_label = dict(zip(labels, lines))  # ensure uniqueness of labels
    fig.legend(by_label.values(), by_label.keys())
    return fig
