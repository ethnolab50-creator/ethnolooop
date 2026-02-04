"""End-to-end RNN pipeline for CIFAR-10 dataset.

This module provides a single function `train_rnn_cifar` that loads
CIFAR-10, builds an RNN (LSTM/GRU), trains it, evaluates on the test set,
and saves the trained model.

The images are reshaped into sequences for RNN processing.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_rnn(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    """Build RNN model for CIFAR-10.
    
    Input images (32x32x3) are reshaped into sequences of timesteps.
    Flattened size: 32*32 = 1024 timesteps, 3 features each.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Reshape (32, 32, 3) -> (1024, 3) for RNN processing
        layers.Reshape((input_shape[0] * input_shape[1], input_shape[2])),
        
        # LSTM layers with dropout
        layers.LSTM(128, activation='relu', return_sequences=True),
        layers.Dropout(0.2),
        
        layers.LSTM(64, activation='relu', return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense classification head
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def train_rnn_cifar(epochs: int = 20,
                    batch_size: int = 128,
                    model_path: str = 'rnn_cifar.h5',
                    verbose: int = 1) -> Tuple[models.Model, dict]:
    """Train an RNN on CIFAR-10 end to end.

    Returns the trained model and a dictionary with test metrics.
    """
    # Load CIFAR-10 (same as CNN)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    num_classes = 10
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    input_shape = x_train.shape[1:]

    model = build_rnn(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ]

    history = model.fit(
        x_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=verbose
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=verbose)

    # Save final model (best model already saved by checkpoint)
    final_path = os.path.splitext(model_path)[0] + '_final.h5'
    model.save(final_path)

    metrics = {'test_loss': float(test_loss), 'test_accuracy': float(test_acc)}

    return model, {'history': history.history, **metrics}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train RNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-path', type=str, default='rnn_cifar.h5')
    args = parser.parse_args()

    model, results = train_rnn_cifar(epochs=args.epochs, batch_size=args.batch_size, model_path=args.model_path)
    print('Test accuracy:', results.get('test_accuracy'))
