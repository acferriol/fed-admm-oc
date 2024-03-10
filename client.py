import flwr as fl
import numpy as np
from model import *


class NodeClient(fl.client.NumPyClient):
    def __init__(self, admm_oc:Admm_OneClassSVM, data):
        self.admm_oc = admm_oc
        self.X = data

    def get_parameters(self, config):
        return get_model_parameters(self.admm_oc)

    def fit(self, parameters, config):
        set_model_parameters(self.admm_oc, parameters)
        train(self.admm_oc, self.X, max_iter=50)
        return get_model_parameters(self.admm_oc)

    def evaluate(self, parameters, config):
        set_model_parameters(self.admm_oc, parameters)
        loss = full_loss(self.X, self.admm_oc.coeff_, self.admm_oc.offset_, self.admm_oc.landa)
        return float(loss)


model = Admm_OneClassSVM(landa=0.02,max_iter=50)
data = generate_data()
transform = Nystroem(gamma=2.0, random_state=None, n_components=200)
transform.fit(data)
X = transform.transform(data)
model = set_initial_parameters(model,X.shape[1])
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=NodeClient(model, X))