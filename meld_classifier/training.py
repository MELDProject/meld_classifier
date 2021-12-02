import logging
import os
from meld_classifier.dataset import Dataset

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger


class Trainer:
    def __init__(self, experiment):
        """
        Class for training neural networks
        """
        self.experiment = experiment
        self.log = logging.getLogger(__name__)

    def init_callbacks(self):
        # tensorboard logs experiment progress
        # tensorboard = TensorBoard(log_dir=os.path.join(self.path,"logs",self.name))
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=self.experiment.network_parameters["max_patience"], verbose=1
        )
        # set model save path
        checkpoint_path = os.path.join(self.experiment.path, "models", self.experiment.name)
        # checkpoint will save only best model, based on val_loss
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            period=1,
        )
        # also log progress to csv file
        csv_logger = CSVLogger(os.path.join(self.experiment.path, "logs", "{}.csv".format(self.experiment.name)))

        return [early_stopping, checkpoint, csv_logger]

    def train(self):
        """Train network to classify vertices into lesional and normal.

        The trained model is saved in experiment_path with experiment_name.
        """
        self.log.info("building model...")
        # saves model in self.experiment.model
        self.experiment.load_model()
        callbacks = self.init_callbacks()

        # get datasets
        self.log.info("loading data...")
        train_dataset = Dataset.from_experiment(self.experiment, mode="train")
        val_dataset = Dataset.from_experiment(self.experiment, mode="val")

        # fit model
        self.log.info("fitting model...")
        self.experiment.model.fit_generator(
            train_dataset,
            steps_per_epoch=len(train_dataset),
            epochs=self.experiment.network_parameters["num_epochs"],
            validation_data=val_dataset,
            validation_steps=len(val_dataset),
            callbacks=callbacks,
            use_multiprocessing=False,
            class_weight=None,
        )
        del train_dataset
        del val_dataset
        self.log.info("...model fitting complete")
