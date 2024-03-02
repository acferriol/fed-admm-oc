import flwr as fl
import utils
from flwr.common import NDArrays,Scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict,Optional,Tuple


def fit_round(server_round: int) -> Dict:
    """Send round number to client"""
    return {"server_round": server_round}

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in
    # `evaluate` itself
    _, (X_test, y_test) = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(server_round:int, parameters: NDArrays, config:Dict[str,Scalar]) -> Optional[Tuple[float,Dict[str,Scalar]]]:
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


model = LogisticRegression()
utils.set_initial_params(model)
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    evaluate_fn=get_evaluate_fn(model),
    on_fit_config_fn=fit_round,
)

history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=20)
)

print(history)
print(model.coef_,model.offset_)


