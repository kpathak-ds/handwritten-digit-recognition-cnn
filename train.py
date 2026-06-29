import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = models.Sequential([

    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32,
        (3,3),
        activation="relu",
    ),

    layers.MaxPooling2D(2,2),

    layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )

])

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

datagen = ImageDataGenerator(

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1

)

history = model.fit(

    datagen.flow(
        x_train,
        y_train,
        batch_size=32
    ),

    epochs=10,

    validation_data=(x_test,y_test)

)

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Accuracy:", accuracy)

os.makedirs("models", exist_ok=True)

model.save("models/digit_model.h5")

print("Model saved successfully")

os.makedirs("outputs", exist_ok=True)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.legend()

plt.savefig(
    "outputs/accuracy_plot.png"
)

plt.show()