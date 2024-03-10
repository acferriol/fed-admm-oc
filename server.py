import flwr as fl
import utils
from flwr.common import NDArrays,Scalar
from sklearn.metrics import *
from typing import Dict,Optional,Tuple
from model import *
from utils import *

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
    model = Admm_OneClassSVM(landa=0.02,max_iter=50)
    
    X_test, y_test = get_test_data()
    transform = Nystroem(gamma=2.0, random_state=None, n_components=200)
    transform.fit(X_test)
    X_test = transform.transform(X_test)

    model = set_initial_parameters(model,X_test.shape[1])

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=20))
