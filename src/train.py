import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Dummy dataset loader (replace with actual dataset)


def load_data():
    X_train = np.random.rand(100, 64, 64, 3)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, 100), 5)
    X_test = np.random.rand(20, 64, 64, 3)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, 5, 20), 5)
    return (X_train, y_train), (X_test, y_test)


# Load dataset
(X_train, y_train), (X_test, y_test) = load_data()

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 gesture classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save
model.save("model.h5")
print("Training complete.")
