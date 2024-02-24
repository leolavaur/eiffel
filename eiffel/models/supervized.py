"""Supervized models for Eiffel."""

from typing import Optional

import keras
from keras.losses import BinaryCrossentropy, Loss
from keras.optimizers import Adam, Optimizer


def mk_popoola_mlp(
    n_features: int,
    loss_fn: Optional[Loss] = None,
    optimizer: Optional[Optimizer] = None,
    learning_rate=0.0001,
) -> keras.Model:
    """Create a MLP model.

    This model is based on the work of Popoola et al. (2021), where it is used on the
    NF-V2 datasets to test new aggregation strategies for FL in heterogeneous intrusion
    detection contexts.

    Parameters
    ----------
    n_features : int
        Number of features in the input data.
    loss_fn : keras.losses.Loss, optional
        Loss function to use. Defaults to `keras.losses.BinaryCrossentropy()`.
    optimizer : keras.optimizers.Optimizer, optional
        Optimizer to use. Defaults to `keras.optimizers.Adam(learning_rate=0.0001)`.
    learning_rate : float, optional
        Learning rate to use with Adam if no optimizer is provided. Defaults to 0.0001.

    Returns
    -------
    keras.Model
        The model.

    References
    ----------
    [1] S. I. Popoola, G. Gui, B. Adebisi, M. Hammoudeh, and H. Gacanin, “Federated Deep
        Learning for Collaborative Intrusion Detection in Heterogeneous Networks,”
        in 2021 IEEE 94th Vehicular Technology Conference (VTC2021-Fall), Sep.
        2021, pp. 1–6. doi: 10.1109/VTC2021-Fall52928.2021.9625505.
    """
    model = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizer or Adam(learning_rate=learning_rate),
        loss=loss_fn or BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model
