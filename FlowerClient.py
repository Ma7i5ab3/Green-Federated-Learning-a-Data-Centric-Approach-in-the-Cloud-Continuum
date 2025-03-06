#import flwr libraries
import flwr as fl
from flwr.common import (
    Status,
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.client import NumPyClient

import tensorflow as tf
tfk = tf.keras
import time
import numpy as np
import pandas as pd
#import codecarbon to evaluate energy consumption
from codecarbon import OfflineEmissionsTracker
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from data_quality import reduce_data_volume, reduce_consistency, reduce_accuracy, reduce_completeness



#Define the CifarClient
class CifarClient(fl.client.Client):
    def __init__(self, cid, model, x_train, y_train, x_test, y_test, dev_energy_consumption = None, dev_emission = None):
        self.cid = cid
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.dev_energy_consumption, self.dev_emission = dev_energy_consumption, dev_emission
        self.measure_parameters  = {
              'effective_epochs': 0,
              'effective_emissions_kg': 0,
              'effective_energy_consumed': 0,
              'effective_duration': 0
        }

    '''
    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")
    '''

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get parameters of the local model."""
        # raise Exception("Not implemented (server-side parameter initialization)")

        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = self.model.get_weights()

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train parameters on the locally held training set."""
        print("Starting Fit")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model parameters
        self.model.set_weights(ndarrays_original)

        # Get hyperparameters for this round
        batch_size: int = ins.config["batch_size"]
        epochs: int = ins.config["local_epochs"]
        #quality_value = ins.config["quality_percentage"] #Only with vertical strategy
        #server_round = ins.config["round"] #Only with vertical strategy
        batch_size = int(min(self.x_train.shape[0] / 10, batch_size))
        print("The batch_size is {}".format(batch_size))
        print("The epochs are {}".format(epochs)) #Only with vertical strategy
        #print("The quality percentage is {}".format(quality_value)) #Only with vertical strategy

        #options = {'data_quality_dimension_percentage': quality_value, 'experiment_method': 'uniform'} #Only with vertical strategy
        #For Vertical Consistency experiments
        #self.x_train, self.y_train = reduce_consistency(self.x_train, self.y_train, options)
        #For Vertical Accuracies experiments
        #elf.x_train, self.y_train = reduce_accuracy(self.x_train, self.y_train, options)
        #For Vertical Completeness experiments
        #self.x_train, self.y_train = reduce_completeness(self.x_train, self.y_train, options)
        print(self.y_train.shape)
        #Remember that for Vertical experiments reduce_consistency doesn't want categorical values
        #self.y_train = tfk.utils.to_categorical(self.y_train)

        #with OfflineEmissionsTracker(country_iso_code="ITA", project_name=1, log_level='warning') as tracker:
        start_time = time.time()
        tracker = OfflineEmissionsTracker()
        tracker.start()
        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=.1,
            callbacks=[
                tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True),
                tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=20, factor=0.5,
                                                min_lr=0.0001),
            ]
        )
        print(tracker)
        tracker.stop()
        self.measure_parameters['effective_duration'] = time.time() - start_time
        print(tracker)
        self.measure_parameters['effective_energy_consumed'] = tracker.final_emissions_data.energy_consumed
        self.measure_parameters['effective_emissions_kg'] = tracker.final_emissions
        self.measure_parameters['effective_epochs'] = len(history.epoch)

        print(self.measure_parameters)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        parameters_updated = ndarrays_to_parameters(parameters_prime)

        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
            "emission_kg": self.measure_parameters['effective_emissions_kg'],
            "energy_consumed": self.measure_parameters['effective_energy_consumed'],
            "duration": self.measure_parameters['effective_duration'],
            "epochs": self.measure_parameters['effective_epochs'],
        }

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_examples_train,
            metrics=results,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate parameters on the locally held test set."""

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model with global parameters
        self.model.set_weights(ndarrays_original)

        # Get config values
        steps: int = ins.config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples_test,
            metrics={"accuracy": float(accuracy)},
        )
    


class FlowerClient(NumPyClient):
    def __init__(self, cid, model, x_train, y_train, x_test, y_test, dev_energy_consumption = None, dev_emission = None):
        self.cid = cid
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.dev_energy_consumption, self.dev_emission = dev_energy_consumption, dev_emission
        self.measure_parameters  = {
              'effective_epochs': 0,
              'effective_emissions_kg': 0,
              'effective_energy_consumed': 0,
              'effective_duration': 0
        }

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        # Get parameters as a list of NumPy ndarray's
        return self.model.get_weights()


    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("Starting Fit")

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        #quality_value = ins.config["quality_percentage"] #Only with vertical strategy
        #server_round = ins.config["round"] #Only with vertical strategy
        batch_size = int(min(self.x_train.shape[0] / 10, batch_size))
        print("The batch_size is {}".format(batch_size))
        print("The epochs are {}".format(epochs)) #Only with vertical strategy
        #print("The quality percentage is {}".format(quality_value)) #Only with vertical strategy

        #options = {'data_quality_dimension_percentage': quality_value, 'experiment_method': 'uniform'} #Only with vertical strategy
        #For Vertical Consistency experiments
        #self.x_train, self.y_train = reduce_consistency(self.x_train, self.y_train, options)
        #For Vertical Accuracies experiments
        #elf.x_train, self.y_train = reduce_accuracy(self.x_train, self.y_train, options)
        #For Vertical Completeness experiments
        #self.x_train, self.y_train = reduce_completeness(self.x_train, self.y_train, options)
        print(self.y_train.shape)
        #Remember that for Vertical experiments reduce_consistency doesn't want categorical values
        #self.y_train = tfk.utils.to_categorical(self.y_train)

        start_time = time.time()
        with OfflineEmissionsTracker(country_iso_code="ITA", project_name=1, log_level='warning') as tracker:
            # Train the model using hyperparameters from config
            history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=.1,
                callbacks=[
                    tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True),
                    tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=20, factor=0.5,
                                                    min_lr=0.0001),
                ]
            )
            tracker.stop()

        #Read emission.csv to extract emissions data
        emissions_df = pd.read_csv('emissions.csv')
        print(f"Energy consumed: {emissions_df['energy_consumed'].iloc[-1]}")
        self.measure_parameters['effective_duration'] = time.time() - start_time
        self.measure_parameters['effective_energy_consumed'] = emissions_df['energy_consumed'].iloc[-1]
        self.measure_parameters['effective_emissions_kg'] = emissions_df['emissions'].iloc[-1]
        self.measure_parameters['effective_epochs'] = len(history.epoch)
        
        
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
            "emission_kg": self.measure_parameters['effective_emissions_kg'],
            "energy_consumed": self.measure_parameters['effective_energy_consumed'],
            "duration": self.measure_parameters['effective_duration'],
            "epochs": self.measure_parameters['effective_epochs'],
        }

        # Build and return response
        return self.model.get_weights(), num_examples_train, results

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        """Evaluate parameters on the locally held test set."""

        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        num_examples_test = len(self.x_test)
        # Build and return response
        return float(loss), num_examples_test, {"accuracy": float(accuracy)}