from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import tensorflow as tf
import numpy as np

from flwr_datasets import FederatedDataset
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
# import tensorflow_privacy


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # if aggregated_parameters is not None:
        #     # Convert `Parameters` to `List[np.ndarray]`
        #     aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

        #     # Save aggregated_ndarrays
        #     print(f"Saving round {server_round} aggregated_ndarrays...")
        #     np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )

    # model = tf.keras.models.load_model('my_model.h5')

    l2_norm_clip = 1.5
    noise_multiplier = 1.3
    num_microbatches = 250
    learning_rate = 0.25

    # optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    # l2_norm_clip=l2_norm_clip,
    # noise_multiplier=noise_multiplier,
    # num_microbatches=num_microbatches,
    # learning_rate=learning_rate)

    # loss = tf.keras.losses.CategoricalCrossentropy(
    # from_logits=True, reduction=tf.losses.Reduction.NONE)

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.3,
    #     fraction_evaluate=0.2,
    #     min_fit_clients=1,
    #     min_evaluate_clients=1,
    #     min_available_clients=1,
    #     evaluate_fn=get_evaluate_fn(model),
    #     on_fit_config_fn=fit_config,
    #     on_evaluate_config_fn=evaluate_config,
    #     initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    # )

    strategy = SaveModelStrategy(
                fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
        # certificates=(
        #     Path(".cache/certificates/ca.crt").read_bytes(),
        #     Path(".cache/certificates/server.pem").read_bytes(),
        #     Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    test = fds.load_full("test")
    test.set_format("numpy")
    x_test, y_test = test["img"] / 255.0, test["label"]

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        # Save aggregated_ndarrays
        print(f"Saving round {server_round} model...")
        model.save(f"server-round-{server_round}-weights.h5")

        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
