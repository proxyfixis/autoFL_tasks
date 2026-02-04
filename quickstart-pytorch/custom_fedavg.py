

from typing import Dict, Optional, Tuple
import numpy as np

from flwr.serverapp.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


class FilteredFedAvg(FedAvg):
    """
    FedAvg + client filtering using:
    - train_loss threshold
    - minimum number of examples
    """

    def __init__(
        self,
        loss_threshold: float = 2.0,
        min_examples: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold
        self.min_examples = min_examples

    # ONLY method overridden
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:

        filtered_results = []

        for client, fit_res in results:
            loss = fit_res.metrics.get("train_loss")
            n = fit_res.num_examples

            if loss is None:
                continue
            if loss >= self.loss_threshold:
                continue
            if n < self.min_examples:
                continue

            filtered_results.append((client, fit_res))

        # IMPORTANT: delegate back to FedAvg for aggregation
        return super().aggregate_fit(
            server_round,
            filtered_results,
            failures,
        )
