#tools for defining network architectures and performance statistics
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import datetime
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import backend as K



def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (total_positives + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    actual_negatives = tf.add(tf.negative(K.clip(y_true, 0, 1)), 1)
    pred_negatives = tf.add(tf.negative(K.clip(y_pred, 0, 1)), 1)
    true_negatives = K.sum(K.round(K.clip(actual_negatives * pred_negatives, 0, 1)))
    specificity = true_negatives / (K.sum(actual_negatives) + K.epsilon())
    return specificity


def focal_loss(gamma=2.0, alpha=0.5):
    """focal loss.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    Args:
        gamma: parameter regulating the influence of easy to predict samples
            gamma=1 results in the cross_entropy case. Higher values give less influence to easy samples
        alpha: class weight for class 1, 1-alpha is the weights for class 0.
    Returns:
        loss_function
    """

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)

        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return focal_loss_fixed


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    References:
        https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)

    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def build_model(
    n_features=0,
    n_neighbours=0,
    network_type="dense",
    layer_sizes=[20],
    shared_layers=[],
    combine_vertices="concat",
    dropout=0,
    learning_rate=0.001,
    loss="binary_crossentropy",
    focal_loss_gamma=2,
    focal_loss_alpha=0.5,
    **kwargs,
):
    """build network model for lesion classification.

    Args:
        n_features
        n_neigbours
        **kwargs: contents of network_parameters
    Returns:
        model: tf.keras.Model
    """
    if network_type == "dense":
        # build dense model
        model = LesionClassifier(layer_sizes, n_features=n_features, n_neighbours=n_neighbours, dropout=dropout)
    elif network_type == "neighbour":
        # build NeighbourLesionClassifier
        model = NeighbourLesionClassifier(
            layer_sizes,
            n_features,
            n_neighbours=n_neighbours,
            shared_layers=shared_layers,
            combine_vertices=combine_vertices,
            dropout=dropout,
        )
    return model.compile_model(
        learning_rate=learning_rate, loss=loss, focal_loss_gamma=focal_loss_gamma, focal_loss_alpha=focal_loss_alpha
    )


import tensorflow as tf


class MCDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)


class LesionClassifier:
    def __init__(self, layer_sizes, n_features, n_neighbours=0, final_activation=tf.nn.sigmoid, dropout=0):
        """build fully connected network model for lesion classification.

        Args:
            layer_sizes (list of int): depth determined by number of sizes
            n_features (int): number of features
            n_neighbours (int): number of neighbours included in the input
            final_activation (function): linear or sigmoid. Linear for regression (which so far hasn't worked)
            dropout (float): dropout value, applied after each layer. Ignored if 0
        """
        # build model architecture
        model = keras.Sequential()
        input_dim = n_features * (n_neighbours + 1)
        for layer in layer_sizes:
            model.add(Dense(layer, input_dim=input_dim))
            model.add(Activation(tf.nn.relu))
            # dropout is always applied but if 0 in dropout or mc_dropout then has no effect.
            model.add(MCDropout(dropout))
            input_dim = layer
        # add final classification/regression layer
        model.add(Dense(1, activation=final_activation))
        self.model = model

    def compile_model(self, learning_rate=0.001, loss="binary_crossentropy", focal_loss_gamma=2, focal_loss_alpha=0.5):
        """complie network model

        Args:
            loss (string): loss function to use. Valid values: focal_loss, soft_f1, and any tf.keras.loss
        """
        # get loss function
        if loss == "focal_loss":
            loss = focal_loss(gamma=focal_loss_gamma, alpha=focal_loss_alpha)
        elif loss == "soft_f1":
            loss = macro_soft_f1
        else:
            # loss is keras loss, can directly plug in model.compile function
            pass
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=loss,
            metrics=["accuracy", precision, sensitivity, specificity],
        )
        return self.model


class NeighbourLesionClassifier(LesionClassifier):
    def __init__(
        self,
        layer_sizes,
        n_features,
        n_neighbours=0,
        shared_layers=[],
        combine_vertices="concat",
        final_activation=tf.nn.sigmoid,
        dropout=0,
    ):
        """build network model for lesion classification with neighborhood.
        Structure: first applies shared layers of size `shared_layers` to each vertex individually,
        then combines vertex representations using the strategy defined in `combine_vertices` (i.e. concat, sum),
        and finally applies fully connected layers defined with `layer_sizes`.

        Args:
            layer_sizes (list of int): depth determined by number of sizes
            n_features (int): number of per-vertex features
            n_neighbours (int): number of neighbours included in the input
            shared_layers (list of int): number of layers that are applied to each vertex individually
            combine_vertices (string): combination strategy of vertex representations (concat or sum)
            final_activation (function): linear or sigmoid. Linear for regression (which so far hasn't worked)
            dropout (float): dropout value, applied after each layer. Ignored if 0
        """
        n_vertices = n_neighbours + 1
        # input to the model
        X_input = tf.keras.layers.Input(n_features * n_vertices)
        # reshape to get (batches, vertices, features)
        X = tf.keras.layers.Reshape([n_vertices, n_features])(X_input)
        # get list of vertices with shape (batches, features)
        X_arr = tf.unstack(X, axis=1)
        # apply shared_layers
        for l in shared_layers:
            dense_layer = Dense(l, activation=tf.nn.relu)
            X_arr = [dense_layer(X) for X in X_arr]
            # dropout is always applied but if 0 in dropout or mc_dropout then has no effect.
            dropout_layer = MCDropout(dropout)
            X_arr = [dropout_layer(X) for X in X_arr]
        # merge combined layers
        if len(X_arr) == 1:
            X = X_arr[0]
        elif combine_vertices == "concat":
            X = tf.keras.layers.concatenate(X_arr)
        elif combine_vertices == "sum":
            X = tf.math.reduce_sum(X_arr, axis=0)
        # apply dense layers
        for l in layer_sizes:
            X = Dense(l, activation=tf.nn.relu)(X)
            X = MCDropout(dropout)(X)
        # add final classification/regression layer
        X = Dense(1, activation=final_activation)(X)

        # build keras model
        model = tf.keras.Model(X_input, X, name="NeighborhoodLesionClassifier")
        self.model = model


def ensemble_models(models, model_input):
    # collect outputs of models in a list
    y_models = [model(model_input) for model in models]
    # averaging outputs
    y_avg = tf.keras.layers.average(y_models)
    # build model from same input and avg output
    model = tf.keras.Model(inputs=model_input, outputs=y_avg, name="ensemble")
    return model
