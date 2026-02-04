"""End-to-end CNN pipeline for CIFAR-10 dataset.

This module provides a single function `train_cnn_cifar` that loads
CIFAR-10, builds a simple CNN, trains it, evaluates on the test set,
and saves the trained model.

Note: I assumed you meant the CIFAR dataset; if you meant a different
"cipher" dataset, tell me and I'll adapt the loader/preprocessing.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def train_cnn_cifar(epochs: int = 20,
                    batch_size: int = 128,
                    model_path: str = 'cnn_cifar.h5',
                    verbose: int = 1) -> Tuple[models.Model, dict]:
    """Train a CNN on CIFAR-10 end to end.

    Returns the trained model and a dictionary with test metrics.
    """
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    num_classes = 10
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    input_shape = x_train.shape[1:]

    model = build_cnn(input_shape, num_classes)
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

    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-path', type=str, default='cnn_cifar.h5')
    args = parser.parse_args()

    model, results = train_cnn_cifar(epochs=args.epochs, batch_size=args.batch_size, model_path=args.model_path)
    print('Test accuracy:', results.get('test_accuracy'))
