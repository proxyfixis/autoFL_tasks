from typing import List, Tuple, Dict, Optional
from flwr.serverapp.strategy import FedAvg
import flwr as fl
from flwr.common import FitRes, Parameters
import numpy as np

import logging
logging.warning("USING MEDIAN AGGREGATION")

class MedianAggregation(fl.server.strategy.FedAvg):
    def aggregate_train(
        self,
        server_round,
        results,
        failures,
    ):
        print(f"[MEDIAN] Round {server_round}, clients={len(results)}")

        if not results:
            return None, {}

        client_weights = []
        for _, fit_res in results:
            client_weights.append(
                fl.common.parameters_to_ndarrays(fit_res.parameters)
            )

        num_layers = len(client_weights[0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_stack = np.stack(
                [client[layer_idx] for client in client_weights],
                axis=0
            )
            aggregated_weights.append(
                np.median(layer_stack, axis=0)
            )

        return (
            fl.common.ndarrays_to_parameters(aggregated_weights),
            {"clients_used": len(client_weights)},
        )
