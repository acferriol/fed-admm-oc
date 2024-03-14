import flwr as fl
import utils
from flwr.common import NDArrays,Scalar
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from typing import Dict,Optional,Tuple
from model import *
from utils import *
from strategy import *
import matplotlib.pyplot as plt
import numpy as np

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: Admm_OneClassSVM):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_model_parameters(model, parameters)
        loss = full_loss(X_test,model.coeff_,model.offset_,model.landa)
        
        pred = model.decision_function(X_test)
        pred = np.array(pred)    
        pred = pred.ravel()
        fpr, tpr, _ = roc_curve(y_test, pred)
        roc_auc = auc(fpr, tpr)
        
        return loss, {"roc_auc": roc_auc}

    return evaluate

if __name__ == "__main__":
    model = Admm_OneClassSVM(landa=0.02,max_iter=150)
    
    X_test_2d, y_test = get_test_data()
    transform = Nystroem(gamma=2.0, random_state=None, n_components=200)
    transform.fit(X_test_2d)
    X_test = transform.transform(X_test_2d)

    model = set_initial_parameters(model,X_test.shape[1])

    pipe_admm = make_pipeline(transform,  model)

    #strategy = fl.server.strategy.FedAvg(
    #    min_available_clients=1,
    #    evaluate_fn=get_evaluate_fn(model),
    #    on_fit_config_fn=fit_round,
    #)
    strategy = ConsensusADMM()
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=20))

    #print(model.coeff_,model.offset_)
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 100), np.linspace(-4.5, 4.5, 100))
    Z_admm = pipe_admm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(9, 6))
    plt.title("One Class ADMM")
    plt.contourf(xx, yy, Z_admm, levels=np.linspace(0, 8, 1), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z_admm, levels=[0], linewidths=2, colors="darkred")
    plt.contourf(xx, yy, Z_admm, levels=[0, Z_admm.max()], colors="palevioletred")
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c="gold", s=18,edgecolors="k")
    plt.show(block=True)