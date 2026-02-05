"""
CNN-LSTM models for attention classification and gaze estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import *

class CNNLSTMModel(keras.Model):
    """CNN-LSTM architecture for temporal eye-gaze analysis"""
    
    def __init__(self, input_shape, num_output_units, task='classification'):
        super(CNNLSTMModel, self).__init__()
        
        self.task = task
        
        # Convolutional layer
        self.conv1d = layers.Conv1D(
            filters=64,
            kernel_size=3,
            strides=2,
            activation='relu',
            padding='same',
            name='conv1d'
        )
        
        self.maxpool = layers.MaxPooling1D(pool_size=2, name='maxpool1d')
        
        # LSTM layers
        self.lstm1 = layers.LSTM(
            units=128,
            return_sequences=True,
            dropout=0.2,
            name='lstm1'
        )
        
        self.lstm2 = layers.LSTM(
            units=64,
            return_sequences=False,
            dropout=0.2,
            name='lstm2'
        )
        
        # Dense layers
        self.dense1 = layers.Dense(units=32, activation='relu', name='dense1')
        self.dropout = layers.Dropout(0.3, name='dropout')
        
        # Output layer
        if task == 'classification':
            self.output_layer = layers.Dense(
                units=num_output_units,
                activation='softmax',
                name='output'
            )
        else:  # regression
            self.output_layer = layers.Dense(
                units=num_output_units,
                activation='linear',
                name='output'
            )
    
    def call(self, inputs, training=False):
        """Forward pass through model"""
        x = self.conv1d(inputs)
        x = self.maxpool(x)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        return outputs


def build_attention_classifier(input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)):
    """Build attention level classification model"""
    model = CNNLSTMModel(
        input_shape=input_shape,
        num_output_units=ATTENTION_NUM_CLASSES,
        task='classification'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_gaze_estimator(input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)):
    """Build gaze point regression model"""
    model = CNNLSTMModel(
        input_shape=input_shape,
        num_output_units=GAZE_OUTPUT_DIM,
        task='regression'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model


if __name__ == "__main__":
    print("Building models...")
    model_attn = build_attention_classifier()
    model_gaze = build_gaze_estimator()
    print("✓ Models built successfully")
