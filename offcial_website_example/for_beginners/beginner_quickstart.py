# -*- coding:utf-8 -*-
""" __official_website__ = https://tensorflow.google.cn/tutorials/quickstart/beginner """

from __future__ import absolute_import, division, print_function, unicode_literals
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # 关掉ssl验证
# - - -

import logging
from log_config import settings

logger = logging.getLogger(__name__)
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    try:
        mnist = tf.keras.datasets.mnist

        # Load and prepare the MNIST dataset.
        # Convert the samples from integers to floating-point numbers:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Build the tf.keras.Sequential model by stacking layers.
        # Choose an optimizer and loss function for training:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        # For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
        predictions = model(x_train[:1]).numpy()
        logger.debug(f"predictions = {predictions}")

        # The tf.nn.softmax function converts these logits to "probabilities" for each class:
        res = tf.nn.softmax(predictions).numpy()
        logger.debug(f"res = {res}")
        logger.debug(f"sum for res = {np.sum(res)}")
        # Note:
        # It is possible to bake this tf.nn.softmax in as the activation function for the last layer of the network.
        # While this can make the model output more directly interpretable,
        # this approach is discouraged as it's impossible to provide an exact
        # and numerically stable loss calculation for all models when using a softmax output.

        # The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index
        # and returns a scalar loss for each example.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # This loss is equal to the negative log probability of the true class:
        # It is zero if the model is sure of the correct class.
        # This untrained model gives probabilities close to random (1/10 for each class),
        # so the initial loss should be close to -tf.log(1/10) ~= 2.3.
        res_loss = loss_fn(y_train[:1], predictions).numpy()
        logger.debug(f"res_loss = {res_loss}")

        logger.debug("1.训练")
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        # The Model.fit method adjusts the model parameters to minimize the loss:
        model.fit(x_train, y_train, epochs=5)  # 训练

        # The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".
        logger.debug("2.评估")
        model.evaluate(x_test, y_test, verbose=2)  # 评估

        # The image classifier is now trained to ~98% accuracy on this dataset.
        # To learn more, read the "TensorFlow tutorials".
        # If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
        logger.debug("3")
        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])
        result = probability_model(x_test[:5])
        logger.debug(f"result =\n {result}")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)
