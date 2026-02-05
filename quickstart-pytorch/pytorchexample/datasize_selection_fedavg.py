# pytorchexample/datasize_selection_fedavg.py

from typing import List, Tuple
from flwr.serverapp.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters


class DataSizeSelectionFedAvg(FedAvg):
    def __init__(self, min_data_size: int, **kwargs):
        super().__init__(**kwargs)
        self.min_data_size = min_data_size
        self.client_data_sizes = {}   # populated lazily
        self.selected_clients_per_round = {}

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, Parameters]]:

        # 🔑 initialize data sizes ONCE, based on real clients
        if not self.client_data_sizes:
            for cid in client_manager.all().keys():
                self.client_data_sizes[cid] = 200 + int(cid) * 300

        selected_clients = []

        for client in client_manager.all().values():
            cid = client.cid
            if self.client_data_sizes[cid] < self.min_data_size:
                continue
            selected_clients.append(client)

        self.selected_clients_per_round[server_round] = [c.cid for c in selected_clients]

        print(
            f"[Round {server_round}] Selected clients (data_size ≥ {self.min_data_size}):",
            self.selected_clients_per_round[server_round],
        )

        return [(c, parameters) for c in selected_clients]
