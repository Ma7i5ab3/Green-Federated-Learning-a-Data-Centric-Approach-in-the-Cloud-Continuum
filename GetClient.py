from FlowerClient import FlowerClient
from data_quality import *
from resnet_model import build_ResNet
import json
import flwr as fl
from flwr.client import Client
from flwr.common import Context
import tensorflow as tf
tfk = tf.keras
from typing import Dict, List, Optional, Tuple, Union
from load_partition import load_partition


# Function to initialize flower node client with its data partition (x_train_p refers to the all x_train set from which the partition is computed)
# Per simulation only data quality dimension (together with or without data volume) can be poisoned 
def get_client_fn_simulation(volume, quality, poisoning_type, x_train_p, y_train_p, x_val_p, y_val_p, indexes_train, indexes_val):
  # It must be called to create an instance of a new FlowerClient
  def client_fn(context: Context) -> FlowerClient:
      """Create a Flower client representing a single organization."""

      # Note: each client gets a different trainloader/valloader, so each client will train and evaluate on their own unique data
      print("Client with CID: {}\n".format((int)(context.node_config["partition-id"])))
      options_volume = {'data_quality_dimension_percentage': volume, 'experiment_method': 'uniform'} # uniform over the classes
      options_quality = {'data_quality_dimension_percentage': quality, 'experiment_method': 'uniform'} # uniform over the classes
      (x_train, y_train), (x_test, y_test) = load_partition((int)(context.node_config["partition-id"]), x_train_p, y_train_p, x_val_p, y_val_p, indexes_train, indexes_val)

      # Data quality poisoning
      x_train, y_train = reduce_data_volume(x_train, y_train, options_volume) #Data Volume can be poisoned alone or together other data quality dimensions
      # Reduce Consistency Horizontally
      if poisoning_type == 'consistency':
        x_train, y_train = reduce_consistency(x_train, y_train, options_quality)
      # Reduce Accuracy Horizontally
      if poisoning_type == 'accuracy':
        x_train, y_train = reduce_accuracy(x_train, y_train, options_quality)
      # Reduce Accuracy Horizontally
      if poisoning_type == 'completeness':
        x_train, y_train = reduce_completeness(x_train, y_train, options_quality)
      # This is just to take the right shape and build the model
      y_train = tfk.utils.to_categorical(y_train)

      # Load model
      model_client = build_ResNet(x_train.shape[1:], y_train.shape[-1])

      # Create a  single Flower client representing a single organization
      return FlowerClient(context.node_config["partition-id"], model_client, x_train, y_train, x_test, y_test)
  return client_fn


def get_evaluate_fn(model, x_test, y_test, file_path):
    """Return an evaluation function for server-side evaluation."""
    print("Evaluate loading")

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)

        with open(file_path, "r") as json_file:
          data = json.load(json_file)

        last_obj = data[-1]
        last_obj[f"Accuracy_{server_round}"] = accuracy

        with open(file_path, "w") as json_file:
          json.dump(data, json_file)

        return loss, {"accuracy": accuracy}

    return evaluate


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 3 if server_round < 4 else 5
    return {"val_steps": val_steps}
