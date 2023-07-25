import tensorflow as tf

class VisioLH:
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(784, activation='relu', input_shape=(784,)),  # Hidden layer with 128 nodes
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 nodes (for 10 classes)
        ])

        return model